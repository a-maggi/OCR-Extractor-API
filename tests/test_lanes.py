import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import pytest

import fakes
from ocr_extractor.lanes import KILL_SIGNAL, ConversionTimeout, Lane, Slot, WorkerCrashed, WorkerUnavailable

process_executor = partial(ProcessPoolExecutor, max_workers=1, mp_context=multiprocessing.get_context("spawn"))


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def noop(*_):
    pass


async def test_timeout_kills_only_the_stuck_worker_and_the_slot_recovers():
    lane = Lane("ocr", 2, process_executor, timeout=1.5)
    await lane.start()
    try:
        stuck_pid, other_pid = [slot._pid for slot in lane._slots]

        stuck = asyncio.create_task(lane.run(fakes.sleep_forever, (), noop, noop))
        await asyncio.sleep(0.2)
        # The other slot keeps working while the first one is stuck
        assert await lane.run(fakes.add, (2, 3), noop, noop) == 5

        with pytest.raises(ConversionTimeout):
            await stuck

        await asyncio.sleep(0.5)
        assert not process_alive(stuck_pid)
        assert process_alive(other_pid)
        # Both slots take work again, one of them on a fresh process
        results = await asyncio.gather(*(lane.run(fakes.add, (i, i), noop, noop) for i in range(4)))
        assert results == [0, 2, 4, 6]
    finally:
        await lane.stop()


async def test_crashed_worker_fails_its_job_and_is_replaced():
    lane = Lane("text", 1, process_executor, timeout=30)
    await lane.start()
    try:
        with pytest.raises(WorkerCrashed):
            await lane.run(fakes.crash, (), noop, noop)
        assert await lane.run(fakes.add, (1, 1), noop, noop) == 2
    finally:
        await lane.stop()


async def test_reports_queue_positions():
    lane = Lane("ocr", 1, process_executor, timeout=30)
    await lane.start()
    try:
        positions: dict[str, list[int]] = {"a": [], "b": [], "c": []}
        tasks = [
            asyncio.create_task(lane.run(fakes.add, (1, 1), positions[name].append, noop)) for name in positions
        ]
        await asyncio.gather(*tasks)
        assert positions["a"][0] == 1
        assert positions["c"][0] == 3
        # c moved up as the jobs ahead of it started
        assert positions["c"][-1] == 1
    finally:
        await lane.stop()


async def test_a_worker_that_died_while_idle_is_replaced_before_the_next_job():
    """The OOM killer (or anything else) kills an idle worker: the next job must still run."""
    lane = Lane("ocr", 1, process_executor, timeout=30)
    await lane.start()
    try:
        slot = lane._slots[0]
        dead_pid = slot._pid
        os.kill(dead_pid, KILL_SIGNAL)
        await asyncio.sleep(1.0)

        results = [await lane.run(fakes.add, (i, 1), noop, noop) for i in range(3)]
        assert results == [1, 2, 3]
        assert slot._pid != dead_pid
    finally:
        await lane.stop()


class FlakyExecutorFactory:
    """Thread executors whose listed creations fail, like a worker initializer that can't start."""

    def __init__(self, failures: set[int]):
        self.failures = failures
        self.created = 0

    def __call__(self) -> ThreadPoolExecutor:
        self.created += 1
        if self.created in self.failures:
            raise RuntimeError(f"initializer failed (attempt {self.created})")
        return ThreadPoolExecutor(max_workers=1)


async def test_starting_a_worker_retries_with_backoff():
    factory = FlakyExecutorFactory(failures={1, 2})
    slot = Slot("ocr-0", factory, max_tasks=100, restart_delays=(0, 0.01, 0.01))
    await slot.start()
    assert factory.created == 3
    assert await slot.run(fakes.add, (2, 2), timeout=5) == 4
    await slot.stop()


async def test_a_worker_that_never_starts_fails_the_job_and_is_retried_on_the_next():
    factory = FlakyExecutorFactory(failures={1, 2, 3})
    slot = Slot("ocr-0", factory, max_tasks=100, restart_delays=(0, 0.01, 0.01))
    with pytest.raises(WorkerUnavailable):
        await slot.run(fakes.add, (1, 1), timeout=5)
    # The next job starts over, and this time the worker comes up
    assert await slot.run(fakes.add, (1, 1), timeout=5) == 2
    await slot.stop()


async def test_a_failed_recycle_keeps_the_result_it_already_had():
    # Worker 1 runs the job; its recycle (worker 2) and both retries fail; the next job's start gets worker 5
    factory = FlakyExecutorFactory(failures={2, 3, 4})
    slot = Slot("ocr-0", factory, max_tasks=1, restart_delays=(0, 0.01, 0.01))
    assert await slot.run(fakes.add, (20, 22), timeout=5) == 42
    await asyncio.sleep(0.1)  # The background restart gives up
    assert not slot.alive
    assert await slot.run(fakes.add, (1, 2), timeout=5) == 3
    await slot.stop()


async def test_cancelling_a_running_job_kills_its_worker_and_the_slot_recovers():
    lane = Lane("ocr", 1, process_executor, timeout=60)
    await lane.start()
    try:
        slot = lane._slots[0]
        busy_pid = slot._pid
        started = asyncio.Event()
        job = asyncio.create_task(lane.run(fakes.sleep_forever, (), noop, started.set))
        await started.wait()
        await asyncio.sleep(0.2)

        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        await asyncio.sleep(0.5)
        assert not process_alive(busy_pid)

        assert await lane.run(fakes.add, (5, 5), noop, noop) == 10
    finally:
        await lane.stop()
