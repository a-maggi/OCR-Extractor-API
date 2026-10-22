import asyncio
import hmac
import json
import logging
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import AsyncIterator, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ocr_extractor import conversion
from ocr_extractor.config import Settings, get_settings
from ocr_extractor.download import download_pdf, host_allowed
from ocr_extractor.jobs import TERMINAL, JobConflict, JobRegistry, Pipeline, QueueFull
from ocr_extractor.lanes import Lane
from ocr_extractor.pdfgen import WARMUP_LINES, text_pdf
from ocr_extractor.triage import TriageLimits

logger = logging.getLogger(__name__)


class JobRequest(BaseModel):
    # The app passes a fresh id per processing attempt, so a retry never collides with an old job
    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    url: str = Field(max_length=2048)
    # Skip the text-layer check and OCR every page ("Retry with OCR")
    force_ocr: bool = False


def process_executor_factory(settings: Settings) -> Callable[[], ProcessPoolExecutor]:
    return partial(
        ProcessPoolExecutor,
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=conversion.worker_init,
        initargs=(
            settings.TORCH_THREADS_PER_WORKER,
            TriageLimits(
                min_page_chars=settings.MIN_PAGE_CHARS,
                image_coverage=settings.IMAGE_COVERAGE_THRESHOLD,
                image_page_min_chars=settings.IMAGE_PAGE_MIN_CHARS,
            ),
        ),
    )


def build_registry(settings: Settings) -> JobRegistry:
    factory = process_executor_factory(settings)
    return JobRegistry(
        text_lane=Lane("text", settings.TEXT_WORKERS, factory, settings.JOB_TIMEOUT_SECONDS),
        ocr_lane=Lane("ocr", settings.OCR_WORKERS, factory, settings.JOB_TIMEOUT_SECONDS),
        pipeline=Pipeline(
            download=partial(
                download_pdf,
                allowed_hosts=settings.allowed_hosts,
                max_bytes=settings.MAX_DOWNLOAD_BYTES,
                timeout=settings.DOWNLOAD_TIMEOUT_SECONDS,
            ),
            text_fn=conversion.text_lane,
            ocr_fn=conversion.ocr_lane,
        ),
        max_queued=settings.MAX_QUEUED_JOBS,
        result_ttl=settings.RESULT_TTL_SECONDS,
    )


async def _warmup(registry: JobRegistry) -> None:
    """One small conversion: brings up the shared layout and OCR-error servers before a real job needs them."""
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "warmup.pdf"
        path.write_bytes(text_pdf(WARMUP_LINES))
        await registry.text_lane.run(registry.pipeline.text_fn, (str(path),), lambda _: None, lambda: None)


async def _start(app: FastAPI, settings: Settings, registry: JobRegistry) -> None:
    """Starts the workers, then warms up. Retries with backoff until it works; /readyz reports the last error."""
    delay = settings.STARTUP_RETRY_SECONDS
    vlm_started = not settings.PREWARM_VLM
    while True:
        try:
            # Lanes first: digital PDFs never need the OCR model, so a failed prewarm mustn't hold them up
            await registry.start()

            if not vlm_started:
                vlm_started = True
                try:
                    from surya.inference import SuryaInferenceManager

                    # Started from the parent so it doesn't belong to any one worker; workers attach to it
                    app.state.vlm = await asyncio.to_thread(SuryaInferenceManager, lazy=False)
                    logger.info("OCR model server is up")
                except Exception:
                    logger.exception("OCR model server failed to start; the first scanned document will retry it")

            if settings.WARMUP:
                await _warmup(registry)
                logger.info("Warmup conversion finished")

            app.state.startup_error = None
            app.state.ready = True
            return
        except Exception as error:
            app.state.startup_error = f"{type(error).__name__}: {error}"
            logger.exception("Startup failed, retrying in %ss", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 300)


def create_app(settings: Settings | None = None, registry: JobRegistry | None = None) -> FastAPI:
    settings = settings or get_settings()
    conversion.configure_environment(settings.TORCH_THREADS_PER_WORKER)
    registry = registry or build_registry(settings)
    expected_auth = f"Bearer {settings.OCR_API_KEY}".encode()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.startup_error = None
        # Startup runs in the background so /healthz answers while workers spawn and models download
        startup = asyncio.create_task(_start(app, settings, registry))
        yield
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)
        await registry.stop()

    app = FastAPI(title="OCR Extractor API", lifespan=lifespan)

    def require_key(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not hmac.compare_digest(authorization.encode(), expected_auth):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz(request: Request):
        if request.app.state.ready:
            return {"ready": True}
        return JSONResponse(
            {"ready": False, "error": request.app.state.startup_error},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.post("/v1/jobs", dependencies=[Depends(require_key)])
    async def create_job(body: JobRequest, request: Request):
        if not host_allowed(body.url, settings.allowed_hosts):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "URL host is not allowed")
        # Not while workers aren't running or startup keeps failing: the job would sit in a queue nothing drains
        startup_error = request.app.state.startup_error
        if not registry.started or startup_error:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"The OCR service isn't ready: {startup_error}" if startup_error else "The OCR service is starting",
                headers={"Retry-After": "15"},
            )
        try:
            job, created = registry.submit(body.job_id, body.url, body.force_ocr)
        except QueueFull as error:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error), headers={"Retry-After": "30"})
        except JobConflict as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error))
        return JSONResponse(
            job.snapshot(include_markdown=False),
            status_code=status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK,
        )

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def get_job(job_id: str):
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        return job.snapshot()

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    async def cancel_job(job_id: str):
        """Drops a queued job or kills the worker converting it, for when the app has given up on it."""
        job = registry.cancel(job_id)
        if not job:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        await asyncio.sleep(0)  # Lets the cancelled job record its final state before we answer
        return job.snapshot(include_markdown=False)

    @app.get("/v1/jobs/{job_id}/events", dependencies=[Depends(require_key)])
    async def job_events(job_id: str):
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

        async def stream() -> AsyncIterator[str]:
            # Subscribe before replaying, so nothing that happens in between is missed
            queue = job.subscribe()
            try:
                yield _sse(job.last_event, job.snapshot())
                if job.finished:
                    return
                while True:
                    try:
                        event, data = await asyncio.wait_for(queue.get(), settings.SSE_PING_SECONDS)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield _sse(event, data)
                    if data["status"] in TERMINAL:
                        return
            finally:
                job.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
