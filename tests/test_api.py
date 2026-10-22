import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

import fakes
from ocr_extractor.api import create_app
from ocr_extractor.jobs import JobRegistry

DIGITAL_URL = "https://files.example.com/f/digital"
SLOW_SCAN_URL = "https://files.example.com/f/scan-slow"


@asynccontextmanager
async def client_for(settings, registry):
    app = create_app(settings, registry)
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if app.state.ready:
                break
            await asyncio.sleep(0.02)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ocr") as client:
            yield client


async def read_events(response: httpx.Response) -> list[tuple[str, dict]]:
    events, event = [], None
    async for line in response.aiter_lines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            events.append((event, json.loads(line[len("data: ") :])))
    return events


async def wait_for_status(client, auth, job_id: str, wanted: str) -> dict:
    for _ in range(200):
        job = (await client.get(f"/v1/jobs/{job_id}", headers=auth)).json()
        if job["status"] == wanted:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {wanted}: {job}")


async def test_health_and_readiness_need_no_key(settings, make_registry):
    async with client_for(settings, make_registry()) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).json() == {"ready": True}


@pytest.mark.parametrize("header", [None, "Bearer wrong-key-0123456789", "test-key-0123456789"])
async def test_job_routes_need_the_api_key(settings, make_registry, header):
    headers = {"Authorization": header} if header else {}
    async with client_for(settings, make_registry()) as client:
        body = {"job_id": "a", "url": DIGITAL_URL}
        assert (await client.post("/v1/jobs", json=body, headers=headers)).status_code == 401
        assert (await client.get("/v1/jobs/a", headers=headers)).status_code == 401
        assert (await client.get("/v1/jobs/a/events", headers=headers)).status_code == 401


async def test_old_marker_endpoint_is_gone(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        response = await client.post("/marker", json={"filepath": "/etc/passwd"}, headers=auth)
        assert response.status_code == 404


@pytest.mark.parametrize("url", ["https://example.com/report.pdf", "http://files.example.com/f/abc", "file:///etc/hosts"])
async def test_rejects_urls_outside_the_allowlist(settings, make_registry, auth, url):
    async with client_for(settings, make_registry()) as client:
        response = await client.post("/v1/jobs", json={"job_id": "a", "url": url}, headers=auth)
        assert response.status_code == 400


async def test_rejects_unsafe_job_ids(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        response = await client.post("/v1/jobs", json={"job_id": "../x", "url": DIGITAL_URL}, headers=auth)
        assert response.status_code == 422


async def test_submit_get_and_dedupe(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        created = await client.post("/v1/jobs", json={"job_id": "doc-1", "url": DIGITAL_URL}, headers=auth)
        assert created.status_code == 202
        assert "markdown" not in created.json()

        again = await client.post("/v1/jobs", json={"job_id": "doc-1", "url": DIGITAL_URL}, headers=auth)
        assert again.status_code == 200
        assert again.json()["job_id"] == "doc-1"

        conflict = await client.post("/v1/jobs", json={"job_id": "doc-1", "url": SLOW_SCAN_URL}, headers=auth)
        assert conflict.status_code == 409

        job = await wait_for_status(client, auth, "doc-1", "done")
        assert job["lane"] == "text"
        assert job["markdown"] == "Invoice total 119.00"
        assert job["pages"] == 1 and job["ocr_pages"] == 0


async def test_unknown_job_is_404(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        assert (await client.get("/v1/jobs/nope", headers=auth)).status_code == 404
        assert (await client.get("/v1/jobs/nope/events", headers=auth)).status_code == 404


async def test_queue_full_is_429(settings, make_registry, auth):
    async with client_for(settings, make_registry(max_queued=1)) as client:
        first = await client.post("/v1/jobs", json={"job_id": "a", "url": SLOW_SCAN_URL}, headers=auth)
        assert first.status_code == 202
        second = await client.post("/v1/jobs", json={"job_id": "b", "url": DIGITAL_URL}, headers=auth)
        assert second.status_code == 429
        assert second.headers["retry-after"] == "30"


async def test_events_replay_a_finished_job_and_close(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        await client.post("/v1/jobs", json={"job_id": "done-job", "url": DIGITAL_URL}, headers=auth)
        await wait_for_status(client, auth, "done-job", "done")

        async with client.stream("GET", "/v1/jobs/done-job/events", headers=auth) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            events = await read_events(response)

        assert [event for event, _ in events] == ["done"]
        assert events[0][1]["markdown"] == "Invoice total 119.00"


async def test_events_stream_a_running_job_to_the_end(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        await client.post("/v1/jobs", json={"job_id": "scan", "url": SLOW_SCAN_URL}, headers=auth)

        async with client.stream("GET", "/v1/jobs/scan/events", headers=auth) as response:
            raw = [line async for line in response.aiter_lines()]

        events = [line[len("event: ") :] for line in raw if line.startswith("event: ")]
        # The first event replays the current state; the stream ends on the terminal event
        assert events[-1] == "done"
        assert "routed" in events or events[0] in {"routed", "converting"}
        assert "converting" in events
        # The 1.5s OCR fake outlasts the 0.2s ping interval
        assert ": ping" in raw
        data = [json.loads(line[len("data: ") :]) for line in raw if line.startswith("data: ")]
        assert data[-1]["lane"] == "ocr" and data[-1]["ocr_pages"] == 1


class RegistryThatFailsToStart(JobRegistry):
    """Its workers fail to start the first `failures` times, like a crashing initializer or llama-server."""

    failures = 1

    async def start(self):
        self.attempts = getattr(self, "attempts", 0) + 1
        if self.attempts <= self.failures:
            raise RuntimeError("worker initializer failed")
        await super().start()


async def wait_for(condition, timeout: float = 5) -> None:
    for _ in range(int(timeout / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


async def test_jobs_are_refused_until_the_workers_start_and_startup_retries(settings, make_registry, auth):
    settings = settings.model_copy(update={"STARTUP_RETRY_SECONDS": 0.2})
    registry = make_registry(registry_class=RegistryThatFailsToStart)
    app = create_app(settings, registry)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ocr") as client:
            await wait_for(lambda: app.state.startup_error is not None)
            ready = await client.get("/readyz")
            assert ready.status_code == 503 and "worker initializer failed" in ready.json()["error"]

            # Nothing would drain the queue: refuse the job instead of accepting it forever
            refused = await client.post("/v1/jobs", json={"job_id": "a", "url": DIGITAL_URL}, headers=auth)
            assert refused.status_code == 503
            assert "worker initializer failed" in refused.json()["detail"]
            assert refused.headers["retry-after"] == "15"

            # The retry starts the workers
            await wait_for(lambda: app.state.ready)
            assert registry.attempts == 2
            accepted = await client.post("/v1/jobs", json={"job_id": "a", "url": DIGITAL_URL}, headers=auth)
            assert accepted.status_code == 202


async def test_a_failed_warmup_is_retried(settings, make_registry, auth):
    fakes.warmup_failures = 2
    settings = settings.model_copy(update={"WARMUP": True, "STARTUP_RETRY_SECONDS": 0.1})
    app = create_app(settings, make_registry())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ocr") as client:
            await wait_for(lambda: app.state.startup_error is not None)
            assert (await client.get("/readyz")).status_code == 503
            await wait_for(lambda: app.state.ready)
            assert fakes.calls["warmup"] == 3
            assert app.state.startup_error is None
            assert (await client.get("/readyz")).json() == {"ready": True}


async def test_delete_cancels_a_job(settings, make_registry, auth):
    async with client_for(settings, make_registry()) as client:
        await client.post("/v1/jobs", json={"job_id": "slow", "url": SLOW_SCAN_URL}, headers=auth)
        await asyncio.sleep(0.2)

        assert (await client.delete("/v1/jobs/slow")).status_code == 401
        cancelled = await client.delete("/v1/jobs/slow", headers=auth)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "failed" and cancelled.json()["error"] == "Cancelled"
        assert (await client.delete("/v1/jobs/unknown", headers=auth)).status_code == 404
