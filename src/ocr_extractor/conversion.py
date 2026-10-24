"""The code that runs inside the worker processes.

Each lane slot is its own spawned process that imports this module. Marker 2.0 workers hold no models: the
layout, OCR-error and OCR (llama-server) models run in shared servers that the first caller starts and every
other process attaches to.
"""

import gc
import os
import re
from pathlib import Path

from ocr_extractor.errors import NoReadableText, OcrServerUnavailable
from ocr_extractor.triage import TriageLimits, looks_empty, triage

_models: dict | None = None
_limits = TriageLimits()

# With paginate_output, Marker puts "{page_id}" and 48 dashes, on their own line, before each page
PAGE_BREAK = re.compile(r"\n*\{(\d+)\}-{48}\n*")


def configure_environment(torch_threads: int) -> None:
    """Called in the parent before any worker is spawned, so workers and the servers they start inherit it."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(torch_threads))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    # Stops surya from starting a multiprocessing pool of its own inside our workers
    os.environ.setdefault("IN_STREAMLIT", "true")
    # Keep llama-server running when the worker that started it exits; the others attach to it
    os.environ.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "true")


def worker_init(torch_threads: int, limits: TriageLimits) -> None:
    import torch
    from marker.models import create_model_dict

    global _models, _limits
    torch.set_num_threads(torch_threads)
    _models = create_model_dict()
    _limits = limits


def split_pages(markdown: str) -> dict[int, str]:
    """Marker's paginated markdown, by page id."""
    parts = PAGE_BREAK.split(markdown)
    pages = {int(page_id): text for page_id, text in zip(parts[1::2], parts[2::2])}
    if parts[0].strip():
        pages[-1] = parts[0]
    return pages


def marker_config(*, disable_ocr: bool, force_ocr: bool) -> dict:
    return {
        "mode": "fast",
        "disable_ocr": disable_ocr,
        "force_ocr": force_ocr,
        "extract_images": False,
        # Page breaks let us check each OCR'd page came back with text; they're removed from the output
        "paginate_output": True,
        # Lab names and notes about decimal separators live in headers and footers
        "keep_pageheader_in_output": True,
        "keep_pagefooter_in_output": True,
        # pdftext starts its own process pool above 1 worker, which deadlocks inside ours
        "pdftext_workers": 1,
        "disable_tqdm": True,
    }


def _convert(path: str, *, disable_ocr: bool, force_ocr: bool) -> dict:
    from marker.converters.pdf import PdfConverter

    converter = PdfConverter(config=marker_config(disable_ocr=disable_ocr, force_ocr=force_ocr), artifact_dict=_models)
    try:
        rendered = converter(path)
        pages = split_pages(rendered.markdown)
        ocr_page_ids = [
            page["page_id"]
            for page in rendered.metadata.get("page_stats", [])
            if page.get("text_extraction_method") == "surya"
        ]
        return {
            "markdown": "\n\n".join(text.strip() for _, text in sorted(pages.items()) if text.strip()),
            "pages": converter.page_count,
            "ocr_pages": len(ocr_page_ids),
            "ocr_pages_without_text": sum(1 for page_id in ocr_page_ids if not pages.get(page_id, "").strip()),
        }
    finally:
        del converter
        gc.collect()


def text_lane(path: str) -> dict:
    """Uses the PDF's own text when it's good. Otherwise returns {"route": "ocr"} so the job moves lanes."""
    result = triage(Path(path), _models["ocr_error_model"], _limits)
    if result.lane == "ocr":
        # Marker keeps a scanner's invisible OCR layer as if it were real text, so OCR every page instead.
        # (Not strip_existing_ocr: Marker 2.0's code for it calls a pypdfium2 method that 5.x removed.)
        return {
            "route": "ocr",
            "reason": result.reason,
            "pages": result.pages,
            "force_ocr": result.replace_ocr_layer,
        }

    converted = _convert(path, disable_ocr=True, force_ocr=False)
    if looks_empty(converted["markdown"], _limits.require_digits):
        # Marker would trust the same text layer again, so OCR every page instead
        return {
            "route": "ocr",
            "reason": "the text layer had no real content",
            "pages": converted["pages"],
            "force_ocr": True,
        }
    return converted


def ocr_lane(path: str, force_ocr: bool) -> dict:
    """Fast mode with OCR on: Marker keeps the good pages' text and OCRs only the scanned or garbled ones."""
    result = _convert(path, disable_ocr=False, force_ocr=force_ocr)

    # Surya turns a failed OCR request (a dead llama-server, a timeout) into empty text and Marker drops it, so a
    # finished conversion can be missing pages. Check the server survived, and that OCR'd pages aren't all empty.
    # Some empty pages are normal: the blank back side of a scanned sheet.
    if not ocr_server_healthy():
        raise OcrServerUnavailable("The OCR model server stopped responding during the conversion")
    if result["ocr_pages"] and result["ocr_pages_without_text"] == result["ocr_pages"]:
        raise OcrServerUnavailable("Every OCR'd page came back empty")
    if looks_empty(result["markdown"], _limits.require_digits):
        raise NoReadableText("OCR found no readable text in the document")
    return result


def ocr_server_healthy() -> bool:
    """True if this worker hasn't used the OCR server, or it still answers. Otherwise forgets the dead server, so
    the next conversion attaches to (or starts) a new one."""
    from surya.inference.backends.spawn import probe_health

    manager = _models["inference_manager"]
    handle = getattr(manager.backend, "handle", None)
    if handle is None:
        return True
    if probe_health(handle.base_url.removesuffix("/v1"), timeout=3.0):
        return True
    manager.stop()
    return False
