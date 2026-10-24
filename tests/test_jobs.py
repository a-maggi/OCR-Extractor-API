import asyncio
import time

import pytest

from ocr_extractor.jobs import JobConflict, JobStatus, QueueFull

BASE = "https://files.example.com/f/"


async def wait_until(condition, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


async def wait_until_finished(registry, job_ids: list[str], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(registry.get(job_id).finished for job_id in job_ids):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"jobs did not finish: {[registry.get(job_id).snapshot() for job_id in job_ids]}")


async def test_a_slow_ocr_job_does_not_block_text_jobs(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        slow, _ = registry.submit("scan", BASE + "scan-slow", force_ocr=False)
        await asyncio.sleep(0.2)
        assert slow.status == JobStatus.CONVERTING and slow.lane == "ocr"

        started = time.monotonic()
        digital = [registry.submit(f"digital-{i}", BASE + f"digital-{i}", False)[0] for i in range(4)]
        await wait_until_finished(registry, [job.id for job in digital])
        elapsed = time.monotonic() - started

        # All four digital jobs finished through the single text slot while the scan was still converting
        assert all(job.status == JobStatus.DONE and job.lane == "text" for job in digital)
        assert slow.status == JobStatus.CONVERTING
        assert elapsed < 1.0

        await wait_until_finished(registry, ["scan"])
        assert slow.status == JobStatus.DONE
        assert slow.ocr_pages == 1
        assert slow.reason == "page 1 has no visible text"
    finally:
        await registry.stop()


async def test_lossy_pages_move_to_ocr_with_the_text_lanes_result_to_splice_into(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        lossy, _ = registry.submit("lossy", BASE + "lossy", force_ocr=False)
        await wait_until_finished(registry, ["lossy"])
        assert lossy.lane == "ocr" and lossy.markdown.endswith("partial=True force_ocr=True")
        assert lossy.reason == "the text lane lost part of the text layer: page 2 kept 78%"
    finally:
        await registry.stop()


@pytest.mark.parametrize("failure", ["timeout", "crash", "dead-server", "unreadable", "bug"])
async def test_a_failed_ocr_move_of_lossy_pages_finishes_with_the_text_lanes_result(make_registry, failure):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("lossy", BASE + f"lossy-{failure}", force_ocr=False)
        await wait_until_finished(registry, ["lossy"])
        # Reported like an OCR move where no page improved, so callers don't OCR it again
        assert job.status == JobStatus.DONE and job.lane == "ocr" and job.ocr_pages == 0
        assert (job.markdown, job.pages) == ("Invoice total 119.00", 2)
        assert job.reason.startswith("the text lane lost part of the text layer: page 2 kept 78%; OCR failed (")
        assert job.reason.endswith("so the text lane's result is kept")
    finally:
        await registry.stop()


async def test_a_failed_scan_still_fails_the_job_instead_of_falling_back(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("scan", BASE + "scan-unreadable", force_ocr=False)
        await wait_until_finished(registry, ["scan"])
        assert job.status == JobStatus.FAILED
    finally:
        await registry.stop()


async def test_cancelling_a_job_during_the_ocr_of_lossy_pages_cancels_it(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("lossy", BASE + "lossy-slow", force_ocr=False)
        await wait_until(lambda: job.lane == "ocr" and job.status == JobStatus.CONVERTING)
        registry.cancel("lossy")
        await wait_until_finished(registry, ["lossy"])
        assert job.status == JobStatus.FAILED and job.error == "Cancelled"
    finally:
        await registry.stop()


async def test_empty_text_layer_output_moves_to_ocr_with_every_page_forced(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        hollow, _ = registry.submit("hollow", BASE + "hollow", force_ocr=False)
        scan, _ = registry.submit("scan", BASE + "scan", force_ocr=False)
        layered, _ = registry.submit("layered", BASE + "scan-layer", force_ocr=False)
        await wait_until_finished(registry, ["hollow", "scan", "layered"])
        assert hollow.lane == "ocr" and hollow.markdown.endswith("force_ocr=True")
        # A plain scan goes through Marker's own per-page choice instead
        assert scan.lane == "ocr" and scan.markdown.endswith("force_ocr=False")
        # A scanner's OCR layer would be kept as if it were real text: every page is OCR'd instead
        assert layered.lane == "ocr" and layered.markdown.endswith("force_ocr=True")
    finally:
        await registry.stop()


async def test_an_ocr_server_blip_is_retried_once(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("blip", BASE + "scan-server-blip", force_ocr=False)
        await wait_until_finished(registry, ["blip"])
        assert job.status == JobStatus.DONE
        assert job.markdown.startswith("Scanned total 119")
    finally:
        await registry.stop()


async def test_an_ocr_server_that_stays_down_fails_the_job(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("down", BASE + "scan-dead-server", force_ocr=False)
        await wait_until_finished(registry, ["down"])
        assert job.status == JobStatus.FAILED
        assert job.error == "The OCR model server isn't responding, try again in a few minutes"
    finally:
        await registry.stop()


async def test_a_scan_with_no_readable_text_fails_instead_of_finishing_empty(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("blank-scan", BASE + "scan-unreadable", force_ocr=False)
        await wait_until_finished(registry, ["blank-scan"])
        assert job.status == JobStatus.FAILED
        assert job.error == "OCR found no readable text in the document"
        assert job.markdown is None
    finally:
        await registry.stop()


async def test_cancel_drops_a_queued_job_and_stops_a_running_one(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        running, _ = registry.submit("running", BASE + "scan-slow-a", force_ocr=False)
        queued, _ = registry.submit("queued", BASE + "scan-slow-b", force_ocr=False)
        await asyncio.sleep(0.3)
        assert running.status == JobStatus.CONVERTING and queued.status == JobStatus.QUEUED

        registry.cancel("queued")
        registry.cancel("running")
        await asyncio.sleep(0.05)
        assert queued.status == JobStatus.FAILED and queued.error == "Cancelled"
        assert running.status == JobStatus.FAILED and running.error == "Cancelled"
        assert registry.ocr_lane.waiting == 0

        # A finished job is left as it is
        done, _ = registry.submit("done", BASE + "digital", force_ocr=False)
        await wait_until_finished(registry, ["done"])
        assert registry.cancel("done").status == JobStatus.DONE
        assert registry.cancel("unknown") is None
    finally:
        await registry.stop()


async def test_force_ocr_skips_the_text_lane(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        job, _ = registry.submit("forced", BASE + "digital", force_ocr=True)
        await wait_until_finished(registry, ["forced"])
        assert job.lane == "ocr"
        assert job.reason == "OCR requested"
    finally:
        await registry.stop()


async def test_same_job_id_returns_the_existing_job(make_registry):
    registry = make_registry()
    await registry.start()
    try:
        first, created = registry.submit("doc-1", BASE + "digital", False)
        again, created_again = registry.submit("doc-1", BASE + "digital", False)
        assert created and not created_again
        assert again is first
        with pytest.raises(JobConflict):
            registry.submit("doc-1", BASE + "other", False)
    finally:
        await registry.stop()


async def test_rejects_jobs_beyond_the_queue_limit(make_registry):
    registry = make_registry(max_queued=2)
    await registry.start()
    try:
        registry.submit("a", BASE + "scan-slow", False)
        registry.submit("b", BASE + "scan-slow", False)
        with pytest.raises(QueueFull):
            registry.submit("c", BASE + "digital", False)
    finally:
        await registry.stop()


async def test_finished_jobs_expire_after_the_ttl(make_registry):
    registry = make_registry()
    registry._result_ttl = 0.1
    await registry.start()
    try:
        registry.submit("short-lived", BASE + "digital", False)
        await wait_until_finished(registry, ["short-lived"])
        await asyncio.sleep(0.2)
        assert registry.get("short-lived") is None
    finally:
        await registry.stop()


async def test_download_errors_fail_the_job(make_registry):
    registry = make_registry()

    async def failing_download(url, dest):
        raise RuntimeError("connection reset")

    registry.pipeline.download = failing_download
    await registry.start()
    try:
        job, _ = registry.submit("broken", BASE + "digital", False)
        await wait_until_finished(registry, ["broken"])
        assert job.status == JobStatus.FAILED
        assert "connection reset" in job.error
    finally:
        await registry.stop()
