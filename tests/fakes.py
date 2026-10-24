"""Stand-ins for the Marker conversions. Module level so process slots can pickle them."""

import os
import time
from pathlib import Path

from ocr_extractor.errors import NoReadableText, OcrServerUnavailable
from ocr_extractor.lanes import ConversionTimeout, WorkerCrashed

# Calls per document, for fakes that fail the first time. Thread slots share it; process slots don't
calls: dict[str, int] = {}
# How many warmup conversions fail before one works (the warmup PDF is the only real PDF the fakes see)
warmup_failures = 0


def fake_text_lane(path: str) -> dict:
    content = Path(path).read_bytes()
    if b"Warmup invoice" in content:
        calls["warmup"] = calls.get("warmup", 0) + 1
        if calls["warmup"] <= warmup_failures:
            raise RuntimeError("layout server not reachable")
    if b"scan" in content:
        # "scan-layer": a scan carrying a scanner's invisible OCR layer, which OCR must replace
        return {"route": "ocr", "reason": "page 1 has no visible text", "pages": 1, "force_ocr": b"layer" in content}
    if b"hollow" in content:
        return {"route": "ocr", "reason": "the text layer had no real content", "pages": 1, "force_ocr": True}
    if b"lossy" in content:
        # Page 2 of 2 lost text; the text lane's own result is what a failed OCR move falls back on
        kept = {"markdown": "Invoice total 119.00", "pages": 2, "ocr_pages": 0}
        return {
            "route": "ocr",
            "reason": "the text lane lost part of the text layer: page 2 kept 78%",
            "pages": 2,
            "force_ocr": True,
            "partial": {"layers": {1: "Invoice total 119.00"}, "text_result": kept},
        }
    return {"markdown": "Invoice total 119.00", "pages": 1, "ocr_pages": 0}


def fake_ocr_lane(path: str, force_ocr: bool, partial: dict | None = None) -> dict:
    content = Path(path).read_bytes()
    calls[content.decode("latin-1")] = calls.get(content.decode("latin-1"), 0) + 1
    if b"slow" in content:
        time.sleep(1.5)
    if b"dead-server" in content:
        raise OcrServerUnavailable("The OCR model server stopped responding during the conversion")
    if b"server-blip" in content and calls[content.decode("latin-1")] == 1:
        raise OcrServerUnavailable("The OCR model server stopped responding during the conversion")
    if b"unreadable" in content:
        raise NoReadableText("OCR found no readable text in the document")
    if b"timeout" in content:
        raise ConversionTimeout("Conversion timed out after 600s")
    if b"crash" in content:
        raise WorkerCrashed("The worker process crashed")
    return {
        "markdown": f"Scanned total 119.00 partial={partial is not None} force_ocr={force_ocr}",
        "pages": 1,
        "ocr_pages": 1,
    }


def fake_pdf_for(url: str) -> bytes:
    """The fake download serves a PDF whose content tells the fake lanes what to do."""
    name = url.rsplit("/", 1)[-1]
    return b"%PDF-1.4 " + name.encode()


async def fake_download(url: str, dest: Path) -> None:
    dest.write_bytes(fake_pdf_for(url))


def sleep_forever() -> None:
    time.sleep(3600)


def crash() -> None:
    os._exit(1)


def add(a: int, b: int) -> int:
    return a + b


def pid() -> int:
    return os.getpid()
