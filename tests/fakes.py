"""Stand-ins for the Marker conversions. Module level so process slots can pickle them."""

import os
import time
from pathlib import Path

from ocr_extractor.errors import NoReadableText, OcrServerUnavailable

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
    return {"markdown": "Invoice total 119.00", "pages": 1, "ocr_pages": 0}


def fake_ocr_lane(path: str, force_ocr: bool) -> dict:
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
    return {"markdown": f"Scanned total 119.00 force_ocr={force_ocr}", "pages": 1, "ocr_pages": 1}


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
