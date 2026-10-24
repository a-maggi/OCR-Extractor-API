"""In-memory job registry. Clients keep the source of truth; a restart here only loses in-flight jobs, and the
client notices (404 on the job) and decides whether to resubmit."""

import asyncio
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable

from ocr_extractor.download import DownloadError
from ocr_extractor.errors import NoReadableText, OcrServerUnavailable
from ocr_extractor.lanes import ConversionTimeout, Lane, WorkerCrashed, WorkerUnavailable

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    DOWNLOADING = "downloading"
    QUEUED = "queued"
    CONVERTING = "converting"
    DONE = "done"
    FAILED = "failed"


TERMINAL = {JobStatus.DONE, JobStatus.FAILED}
OCR_FAILURES = (ConversionTimeout, WorkerCrashed, WorkerUnavailable, OcrServerUnavailable, NoReadableText)


class QueueFull(Exception):
    pass


class JobConflict(Exception):
    pass


@dataclass
class Job:
    id: str
    url: str
    force_ocr: bool
    status: JobStatus = JobStatus.DOWNLOADING
    lane: str | None = None
    position: int | None = None
    # Why the job is in the OCR lane
    reason: str | None = None
    pages: int | None = None
    ocr_pages: int | None = None
    markdown: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    last_event: str = "downloading"
    cancel_requested: bool = field(default=False, repr=False)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def snapshot(self, include_markdown: bool = True) -> dict:
        data = {
            "job_id": self.id,
            "status": self.status.value,
            "lane": self.lane,
            "position": self.position,
            "reason": self.reason,
            "pages": self.pages,
            "ocr_pages": self.ocr_pages,
            "error": self.error,
            "duration_seconds": round(self.finished_at - self.created_at, 2) if self.finished_at else None,
        }
        if include_markdown:
            data["markdown"] = self.markdown
        return data

    def update(self, event: str, **changes: Any) -> None:
        for name, value in changes.items():
            setattr(self, name, value)
        self.last_event = event
        snapshot = self.snapshot()
        for queue in self._subscribers:
            queue.put_nowait((event, snapshot))

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)


@dataclass
class Pipeline:
    """The pieces a job runs through. Tests swap in fakes; text_fn and ocr_fn must be picklable for process slots."""

    download: Callable[[str, Path], Awaitable[None]]
    text_fn: Callable[[str], dict]
    # (path, force_ocr, partial): partial is the text lane's request to OCR only some pages, or None
    ocr_fn: Callable[[str, bool, dict | None], dict]


class JobRegistry:
    def __init__(
        self,
        *,
        text_lane: Lane,
        ocr_lane: Lane,
        pipeline: Pipeline,
        max_queued: int,
        result_ttl: float,
    ):
        self.text_lane = text_lane
        self.ocr_lane = ocr_lane
        self.pipeline = pipeline
        self._max_queued = max_queued
        self._result_ttl = result_ttl
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._sweeper: asyncio.Task | None = None
        self.started = False

    async def start(self) -> None:
        """Starts the lanes' workers. Safe to call again after a failure."""
        await asyncio.gather(self.text_lane.start(), self.ocr_lane.start())
        if not self._sweeper:
            self._sweeper = asyncio.create_task(self._sweep_forever())
        self.started = True

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(self.text_lane.stop(), self.ocr_lane.stop())

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job and self._expired(job):
            del self._jobs[job_id]
            return None
        return job

    def submit(self, job_id: str, url: str, force_ocr: bool) -> tuple[Job, bool]:
        """Returns (job, created). The same job_id again returns the existing job instead of a second run."""
        existing = self.get(job_id)
        if existing:
            if existing.url != url:
                raise JobConflict(f"Job {job_id} already exists for a different URL")
            return existing, False

        unfinished = sum(1 for job in self._jobs.values() if not job.finished)
        if unfinished >= self._max_queued:
            raise QueueFull(f"{unfinished} jobs are already waiting")

        job = Job(id=job_id, url=url, force_ocr=force_ocr)
        self._jobs[job_id] = job
        task = asyncio.create_task(self._run(job))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return job, True

    def cancel(self, job_id: str) -> Job | None:
        """Stops a job: drops it from its queue, or kills the worker converting it. A finished job is left as is."""
        job = self.get(job_id)
        if job and not job.finished and (task := self._tasks.get(job_id)):
            job.cancel_requested = True
            task.cancel()
        return job

    async def _run(self, job: Job) -> None:
        workdir = Path(tempfile.mkdtemp(prefix="ocr-job-"))
        path = workdir / "document.pdf"
        try:
            await self.pipeline.download(job.url, path)

            if job.force_ocr:
                result = await self._ocr(job, path, force_ocr=True, reason="OCR requested")
            else:
                result = await self._convert(job, self.text_lane, self.pipeline.text_fn, (str(path),))
                if result.get("route") == "ocr":
                    result = await self._move_to_ocr(job, path, result)

            job.update(
                "done",
                status=JobStatus.DONE,
                position=None,
                markdown=result["markdown"],
                pages=result["pages"],
                ocr_pages=result["ocr_pages"],
                finished_at=time.monotonic(),
            )
            logger.info(
                "job %s done: lane=%s pages=%s ocr_pages=%s in %ss",
                job.id,
                job.lane,
                job.pages,
                job.ocr_pages,
                job.snapshot()["duration_seconds"],
            )
        except asyncio.CancelledError:
            if job.cancel_requested:
                self._fail(job, "Cancelled")
                return
            self._fail(job, "The OCR service shut down")
            raise
        except OcrServerUnavailable:
            self._fail(job, "The OCR model server isn't responding, try again in a few minutes")
        except (DownloadError, ConversionTimeout, WorkerCrashed, WorkerUnavailable, NoReadableText) as error:
            self._fail(job, str(error))
        except Exception as error:
            logger.exception("job %s failed", job.id)
            self._fail(job, f"{type(error).__name__}: {error}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _move_to_ocr(self, job: Job, path: Path, route: dict) -> dict:
        """The text lane's one move to OCR. If it fails and the text lane kept a result, that result is the answer."""
        partial = route.get("partial")
        force_ocr = bool(route["force_ocr"])
        try:
            return await self._ocr(job, path, force_ocr=force_ocr, reason=route["reason"], partial=partial)
        except OCR_FAILURES as error:
            if not partial:
                raise
            logger.warning("job %s: OCR failed (%s), keeping the text lane's result", job.id, error)
            reason = f"{route['reason']}; OCR failed ({error}), so the text lane's result is kept"
            job.update("routed", lane=self.text_lane.name, reason=reason)
            return partial["text_result"]

    async def _ocr(self, job: Job, path: Path, *, force_ocr: bool, reason: str, partial: dict | None = None) -> dict:
        args = (str(path), force_ocr, partial)
        try:
            return await self._convert(job, self.ocr_lane, self.pipeline.ocr_fn, args, reason)
        except OcrServerUnavailable as error:
            # The worker dropped its connection to the dead server; the retry attaches to (or starts) a new one
            logger.warning("job %s: %s, retrying once", job.id, error)
            return await self._convert(job, self.ocr_lane, self.pipeline.ocr_fn, args, reason)

    async def _convert(self, job: Job, lane: Lane, fn: Callable, args: tuple, reason: str | None = None) -> dict:
        moved = job.lane is not None and job.lane != lane.name
        job.update(
            "routed" if moved else "queued",
            status=JobStatus.QUEUED,
            lane=lane.name,
            reason=reason or job.reason,
        )
        return await lane.run(
            fn,
            args,
            on_position=lambda position: job.update("queued", status=JobStatus.QUEUED, position=position),
            on_start=lambda: job.update("converting", status=JobStatus.CONVERTING, position=None),
        )

    def _fail(self, job: Job, error: str) -> None:
        job.update("failed", status=JobStatus.FAILED, position=None, error=error, finished_at=time.monotonic())
        logger.warning("job %s failed: %s", job.id, error)

    def _expired(self, job: Job) -> bool:
        return job.finished_at is not None and time.monotonic() - job.finished_at > self._result_ttl

    async def _sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, self._result_ttl))
            for job_id in [job_id for job_id, job in self._jobs.items() if self._expired(job)]:
                del self._jobs[job_id]
