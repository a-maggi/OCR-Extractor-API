import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import fakes
from fakes import fake_download, fake_ocr_lane, fake_text_lane
from ocr_extractor.config import Settings
from ocr_extractor.jobs import JobRegistry, Pipeline
from ocr_extractor.lanes import Lane
from ocr_extractor.pdfgen import WARMUP_LINES, text_pdf

API_KEY = "test-key-0123456789"


@pytest.fixture
def digital_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "digital.pdf"
    path.write_bytes(text_pdf(WARMUP_LINES))
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """An image-only PDF, like a scan: the words are pixels, there is no text layer."""
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(WARMUP_LINES):
        draw.text((120, 160 + row * 60), line, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PDF", resolution=150)
    path = tmp_path / "scanned.pdf"
    path.write_bytes(buffer.getvalue())
    return path


@pytest.fixture
def ocr_layer_pdf(tmp_path: Path) -> Path:
    """Text in invisible render mode only: what a scanner's searchable-PDF option produces."""
    path = tmp_path / "ocr-layer.pdf"
    path.write_bytes(text_pdf(WARMUP_LINES, invisible=True))
    return path


@pytest.fixture
def settings() -> Settings:
    return Settings(
        OCR_API_KEY=API_KEY,
        ALLOWED_SOURCE_HOSTS="files.example.com,*.cdn.example.net",
        TEXT_WORKERS=1,
        OCR_WORKERS=1,
        MAX_QUEUED_JOBS=5,
        WARMUP=False,
        SSE_PING_SECONDS=0.2,
    )


def thread_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def make_registry(settings: Settings):
    def make(
        text_workers: int = 1,
        ocr_workers: int = 1,
        max_queued: int | None = None,
        registry_class: type[JobRegistry] = JobRegistry,
    ) -> JobRegistry:
        return registry_class(
            text_lane=Lane("text", text_workers, thread_executor, timeout=10),
            ocr_lane=Lane("ocr", ocr_workers, thread_executor, timeout=10),
            pipeline=Pipeline(download=fake_download, text_fn=fake_text_lane, ocr_fn=fake_ocr_lane),
            max_queued=max_queued or settings.MAX_QUEUED_JOBS,
            result_ttl=settings.RESULT_TTL_SECONDS,
        )

    return make


@pytest.fixture(autouse=True)
def reset_fakes():
    fakes.calls.clear()
    fakes.warmup_failures = 0


@pytest.fixture
def auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}

