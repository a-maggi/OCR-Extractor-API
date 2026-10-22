"""A lane is a FIFO queue drained by N slots. Each slot is a single-process executor.

One process per slot, rather than one pool per lane, lets a stuck conversion be killed without taking down the
conversions running next to it.
"""

import asyncio
import logging
import os
import signal
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import psutil

logger = logging.getLogger(__name__)

KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
# Waits before each attempt to start a worker process
RESTART_DELAYS = (0.0, 2.0, 10.0)


class ConversionTimeout(Exception):
    pass


class WorkerCrashed(Exception):
    pass


class WorkerUnavailable(Exception):
    """The worker process couldn't be started."""


class Slot:
    def __init__(
        self,
        name: str,
        executor_factory: Callable[[], Executor],
        max_tasks: int,
        restart_delays: Sequence[float] = RESTART_DELAYS,
    ):
        self.name = name
        self._executor_factory = executor_factory
        self._max_tasks = max_tasks
        self._restart_delays = restart_delays
        self._executor: Executor | None = None
        self._pid: int | None = None
        self._tasks = 0
        self._start_lock = asyncio.Lock()
        self._background_start: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        if self._executor is None:
            return False
        # ProcessPoolExecutor flags itself broken once its worker dies, even while idle
        if getattr(self._executor, "_broken", False):
            return False
        return self._pid is None or psutil.pid_exists(self._pid)

    async def start(self) -> None:
        """Starts the worker if it isn't running, retrying with backoff. Raises WorkerUnavailable if every try fails."""
        async with self._start_lock:
            if self.alive:
                return
            error: Exception | None = None
            for delay in self._restart_delays:
                if delay:
                    await asyncio.sleep(delay)
                self._discard()
                try:
                    self._executor = self._executor_factory()
                    self._tasks = 0
                    if isinstance(self._executor, ProcessPoolExecutor):
                        # Starts the process (and runs the initializer) now, and gives us the pid to kill later
                        self._pid = await asyncio.get_running_loop().run_in_executor(self._executor, os.getpid)
                    return
                except Exception as start_error:
                    error = start_error
                    logger.warning("%s: worker failed to start: %s", self.name, start_error)
            self._discard()
            raise WorkerUnavailable(f"The worker process failed to start: {error}") from error

    async def stop(self) -> None:
        if self._background_start:
            self._background_start.cancel()
        self._discard()

    async def run(self, fn: Callable[..., Any], args: tuple, timeout: float) -> Any:
        # Replaces a worker that died while idle (OOM killer, a failed restart) before handing it the job
        await self.start()
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, fn, *args)
        except BrokenProcessPool:
            # It died between the check and the submit: not this job's fault, so try once more on a new worker
            self._discard()
            await self.start()
            future = loop.run_in_executor(self._executor, fn, *args)

        try:
            result = await asyncio.wait_for(future, timeout)
        except TimeoutError:
            logger.warning("%s: conversion timed out after %ss, restarting the worker", self.name, timeout)
            self._restart_soon()
            raise ConversionTimeout(f"Conversion timed out after {timeout:.0f}s") from None
        except BrokenProcessPool:
            logger.error("%s: worker process died, restarting it", self.name)
            self._restart_soon()
            raise WorkerCrashed("The worker process crashed") from None
        except asyncio.CancelledError:
            # The job was cancelled, or the lane is stopping: stop the conversion by killing its worker
            self._discard()
            raise

        self._tasks += 1
        if self._tasks >= self._max_tasks:
            # Recycle long-lived workers so slow leaks (pdfium, torch) don't build up. The result is already in
            # hand: if the new worker fails to start, the next run tries again
            self._restart_soon()
        return result

    def _restart_soon(self) -> None:
        """Kills the current worker and starts a new one in the background, so it's ready for the next job."""
        self._discard()

        async def start_quietly() -> None:
            try:
                await self.start()
            except WorkerUnavailable as error:
                logger.error("%s: %s", self.name, error)

        self._background_start = asyncio.create_task(start_quietly())

    def _discard(self) -> None:
        if self._pid is not None:
            try:
                os.kill(self._pid, KILL_SIGNAL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            self._pid = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


@dataclass
class _Work:
    fn: Callable[..., Any]
    args: tuple
    future: asyncio.Future
    on_position: Callable[[int], None]
    on_start: Callable[[], None]


class Lane:
    def __init__(
        self,
        name: str,
        size: int,
        executor_factory: Callable[[], Executor],
        timeout: float,
        max_tasks_per_worker: int = 200,
        restart_delays: Sequence[float] = RESTART_DELAYS,
    ):
        self.name = name
        self.timeout = timeout
        self._slots = [
            Slot(f"{name}-{i}", executor_factory, max_tasks_per_worker, restart_delays) for i in range(size)
        ]
        self._queue: asyncio.Queue[_Work] = asyncio.Queue()
        self._waiting: list[_Work] = []
        self._consumers: list[asyncio.Task] = []

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    async def start(self) -> None:
        """Starts the slots' workers. Safe to call again after a failure: it only starts what isn't running."""
        if not self._consumers:
            self._consumers = [asyncio.create_task(self._consume(slot)) for slot in self._slots]
        results = await asyncio.gather(*(slot.start() for slot in self._slots), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]

    async def stop(self) -> None:
        for consumer in self._consumers:
            consumer.cancel()
        await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers = []
        await asyncio.gather(*(slot.stop() for slot in self._slots))

    async def run(
        self,
        fn: Callable[..., Any],
        args: tuple,
        on_position: Callable[[int], None],
        on_start: Callable[[], None],
    ) -> Any:
        """Waits for a free slot, then runs fn(*args) in it. Positions are 1-based. Cancelling stops the conversion."""
        work = _Work(fn, args, asyncio.get_running_loop().create_future(), on_position, on_start)
        self._waiting.append(work)
        on_position(len(self._waiting))
        self._queue.put_nowait(work)
        try:
            return await work.future
        finally:
            if work in self._waiting:  # Cancelled while still queued
                self._waiting.remove(work)
                self._report_positions()

    def _report_positions(self) -> None:
        for position, work in enumerate(self._waiting, start=1):
            work.on_position(position)

    async def _consume(self, slot: Slot) -> None:
        while True:
            work = await self._queue.get()
            if work.future.done():
                continue
            self._waiting.remove(work)
            self._report_positions()
            work.on_start()

            attempt = asyncio.create_task(slot.run(work.fn, work.args, self.timeout))
            # Cancelling the job cancels its conversion, which kills the worker running it
            work.future.add_done_callback(lambda future, attempt=attempt: future.cancelled() and attempt.cancel())
            try:
                result = await attempt
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling() == 0:
                    continue  # Only the job was cancelled; the slot starts a new worker on its next run
                attempt.cancel()
                raise
            except Exception as error:
                if not work.future.done():
                    work.future.set_exception(error)
                continue
            if not work.future.done():
                work.future.set_result(result)
