"""The OCR lane's checks on a finished conversion, with Marker replaced by a fake."""

from dataclasses import dataclass

import pytest

from ocr_extractor import conversion
from ocr_extractor.errors import NoReadableText, OcrServerUnavailable
from ocr_extractor.triage import TriageLimits, TriageResult


def fake_convert(markdown: str, *, ocr_pages: int, ocr_pages_without_text: int):
    def convert(path: str, *, disable_ocr: bool, force_ocr: bool) -> dict:
        return {
            "markdown": markdown,
            "pages": 2,
            "ocr_pages": ocr_pages,
            "ocr_pages_without_text": ocr_pages_without_text,
        }

    return convert


@pytest.mark.parametrize("lane", [{"disable_ocr": True, "force_ocr": False}, {"disable_ocr": False, "force_ocr": True}])
def test_both_lanes_keep_page_headers_and_footers(lane):
    config = conversion.marker_config(**lane)
    assert config["keep_pageheader_in_output"] and config["keep_pagefooter_in_output"]
    assert config["disable_ocr"] == lane["disable_ocr"] and config["force_ocr"] == lane["force_ocr"]


@pytest.fixture
def healthy(monkeypatch):
    state = {"healthy": True}
    monkeypatch.setattr(conversion, "ocr_server_healthy", lambda: state["healthy"])
    return state


def test_a_good_ocr_result_passes(monkeypatch, healthy):
    monkeypatch.setattr(conversion, "_convert", fake_convert("Scanned total 119.00", ocr_pages=2, ocr_pages_without_text=0))
    assert conversion.ocr_lane("doc.pdf", False)["markdown"] == "Scanned total 119.00"


def test_a_blank_scanned_page_is_fine_when_the_rest_has_text(monkeypatch, healthy):
    monkeypatch.setattr(conversion, "_convert", fake_convert("Scanned total 119.00", ocr_pages=2, ocr_pages_without_text=1))
    assert conversion.ocr_lane("doc.pdf", False)["ocr_pages"] == 2


def test_a_dead_ocr_server_fails_the_conversion_even_with_some_text(monkeypatch, healthy):
    """Surya turns failed requests into empty text, so a finished conversion can be missing pages."""
    monkeypatch.setattr(conversion, "_convert", fake_convert("Scanned total 119.00", ocr_pages=2, ocr_pages_without_text=1))
    healthy["healthy"] = False
    with pytest.raises(OcrServerUnavailable):
        conversion.ocr_lane("doc.pdf", False)


def test_every_ocrd_page_empty_is_treated_as_a_server_failure(monkeypatch, healthy):
    monkeypatch.setattr(conversion, "_convert", fake_convert("Northwind 2026", ocr_pages=2, ocr_pages_without_text=2))
    with pytest.raises(OcrServerUnavailable, match="Every OCR'd page came back empty"):
        conversion.ocr_lane("doc.pdf", False)


def test_empty_output_from_a_healthy_server_means_no_readable_text(monkeypatch, healthy):
    monkeypatch.setattr(conversion, "_convert", fake_convert("", ocr_pages=0, ocr_pages_without_text=0))
    with pytest.raises(NoReadableText):
        conversion.ocr_lane("doc.pdf", True)


def test_number_free_output_passes_unless_digits_are_required(monkeypatch, healthy):
    letter = "Dear customer, thank you for your order"
    monkeypatch.setattr(conversion, "_convert", fake_convert(letter, ocr_pages=1, ocr_pages_without_text=0))
    assert conversion.ocr_lane("doc.pdf", False)["markdown"] == letter

    monkeypatch.setattr(conversion, "_limits", TriageLimits(require_digits=True))
    with pytest.raises(NoReadableText):
        conversion.ocr_lane("doc.pdf", False)


def test_text_lane_moves_number_free_output_to_ocr_only_when_digits_are_required(monkeypatch, digital_pdf):
    monkeypatch.setattr(conversion, "triage", lambda path, model, limits: TriageResult("text", "text layer looks good", 1))
    monkeypatch.setattr(conversion, "_models", {"ocr_error_model": None})
    letter = "Dear customer, thank you for your order"
    monkeypatch.setattr(conversion, "_convert", fake_convert(letter, ocr_pages=0, ocr_pages_without_text=0))
    assert conversion.text_lane(str(digital_pdf))["markdown"] == letter

    monkeypatch.setattr(conversion, "_limits", TriageLimits(require_digits=True))
    assert conversion.text_lane(str(digital_pdf))["route"] == "ocr"


@dataclass
class FakeHandle:
    base_url: str


class FakeBackend:
    def __init__(self, handle):
        self.handle = handle


class FakeManager:
    def __init__(self, handle):
        self.backend = FakeBackend(handle)
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.backend.handle = None


def test_health_check_is_skipped_when_the_worker_never_used_the_ocr_server(monkeypatch):
    manager = FakeManager(handle=None)
    monkeypatch.setattr(conversion, "_models", {"inference_manager": manager})
    assert conversion.ocr_server_healthy()
    assert not manager.stopped


def test_a_dead_ocr_server_is_forgotten_so_the_next_conversion_reconnects(monkeypatch):
    # Nothing listens on port 9 (discard); the probe fails fast
    manager = FakeManager(handle=FakeHandle(base_url="http://127.0.0.1:9/v1"))
    monkeypatch.setattr(conversion, "_models", {"inference_manager": manager})
    assert not conversion.ocr_server_healthy()
    assert manager.stopped and manager.backend.handle is None
