"""The OCR lane's checks on a finished conversion, with Marker replaced by a fake."""

from dataclasses import dataclass

import pytest

from ocr_extractor import conversion
from ocr_extractor.errors import NoReadableText, OcrServerUnavailable
from ocr_extractor.triage import TriageLimits, TriageResult


def fake_convert(markdown: str, *, ocr_pages: int, ocr_pages_without_text: int):
    def convert(path: str, *, disable_ocr: bool, force_ocr: bool, page_range: list[int] | None = None) -> dict:
        return {
            "markdown": markdown,
            "page_markdown": {0: markdown},
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


WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()


def layer(*, words: int = 10) -> str:
    return " ".join(WORDS[:words])


def scripted_convert(text_pages: dict[int, str], ocr_pages: dict[int, str] | None = None, calls: list | None = None):
    """Marker with OCR off returns text_pages. With OCR on, it returns ocr_pages within the requested page_range."""

    def convert(path: str, *, disable_ocr: bool, force_ocr: bool, page_range: list[int] | None = None) -> dict:
        if calls is not None:
            calls.append({"disable_ocr": disable_ocr, "force_ocr": force_ocr, "page_range": page_range})
        pages = text_pages if disable_ocr else {i: t for i, t in (ocr_pages or {}).items() if i in (page_range or [i])}
        return {
            "markdown": conversion.join_pages(pages),
            "page_markdown": pages,
            "pages": len(pages),
            "ocr_pages": 0 if disable_ocr else len(pages),
            "ocr_pages_without_text": 0 if disable_ocr else sum(1 for text in pages.values() if not text.strip()),
        }

    return convert


def text_lane_over(
    monkeypatch, digital_pdf, layers: list[str], kept: list[str], min_coverage: float | None = 0.9, **limits
) -> dict:
    """Runs the text lane on a document whose pages have these text layers and whose conversion kept `kept`."""
    if min_coverage is not None:
        monkeypatch.setattr(conversion, "_limits", TriageLimits(min_text_coverage=min_coverage, **limits))
    monkeypatch.setattr(conversion, "_models", {"ocr_error_model": None})
    result = TriageResult("text", "text layer looks good", len(layers), page_texts=layers)
    monkeypatch.setattr(conversion, "triage", lambda path, model, limits: result)
    monkeypatch.setattr(conversion, "_convert", scripted_convert(dict(enumerate(kept))))
    return conversion.text_lane(str(digital_pdf))


def test_text_lane_reroutes_only_the_pages_that_lost_text(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()] * 3, [layer(), layer(words=5), layer()])
    assert result["route"] == "ocr" and result["force_ocr"] and result["pages"] == 3
    assert result["reason"] == "the text lane lost part of the text layer: page 2 kept 50%"
    assert result["partial"]["layers"] == {1: layer()}
    assert result["partial"]["text_result"]["page_markdown"] == {0: layer(), 1: layer(words=5), 2: layer()}


def test_the_reason_names_every_lossy_page(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()] * 3, [layer(words=3), layer(), layer(words=7)])
    assert result["reason"] == "the text lane lost part of the text layer: page 1 kept 30%, page 3 kept 70%"


def test_a_one_page_document_reroutes_its_page(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()], [layer(words=4)])
    assert result["partial"]["layers"] == {0: layer()}


def test_a_document_where_every_page_is_lossy_reroutes_every_page(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()] * 2, [layer(words=3)] * 2)
    assert sorted(result["partial"]["layers"]) == [0, 1]


def test_a_page_exactly_at_the_threshold_stays_in_the_text_lane(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()], [layer(words=8)], min_coverage=0.8)
    assert "route" not in result
    assert text_lane_over(monkeypatch, digital_pdf, [layer()], [layer(words=8)], min_coverage=0.81)["route"] == "ocr"


def test_a_page_that_lost_a_single_word_stays_in_the_text_lane(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()], [layer(words=9)], min_coverage=0.99)
    assert "route" not in result


def test_only_the_lowest_scoring_pages_are_ocrd_up_to_the_cap(monkeypatch, digital_pdf):
    kept = [layer(words=4), layer(words=2), layer(words=6), layer(words=3), layer()]
    result = text_lane_over(monkeypatch, digital_pdf, [layer()] * 5, kept, max_rerouted_pages=2)
    assert sorted(result["partial"]["layers"]) == [1, 3]
    assert result["reason"] == (
        "the text lane lost part of the text layer: page 2 kept 20%, page 4 kept 30%"
        "; pages 1, 3 also lost text, not OCR'd (limit: 2 per job)"
    )


def test_one_page_over_the_cap_is_named_in_the_singular(monkeypatch, digital_pdf):
    kept = [layer(words=2), layer(words=3)]
    result = text_lane_over(monkeypatch, digital_pdf, [layer()] * 2, kept, max_rerouted_pages=1)
    assert result["reason"].endswith("; page 2 also lost text, not OCR'd (limit: 1 per job)")


def test_the_default_threshold_reroutes_a_page_that_lost_three_words_in_a_hundred(monkeypatch, digital_pdf):
    hundred = [a + b for a in "abcdefghij" for b in "klmnopqrst"]
    layers = [" ".join(hundred)]
    monkeypatch.setattr(conversion, "_limits", TriageLimits())
    for lost, routed in ((1, False), (3, True)):
        result = text_lane_over(monkeypatch, digital_pdf, layers, [" ".join(hundred[lost:])], min_coverage=None)
        assert (result.get("route") == "ocr") is routed


def test_a_zero_threshold_turns_the_coverage_check_off(monkeypatch, digital_pdf):
    result = text_lane_over(monkeypatch, digital_pdf, [layer()], [layer(words=3)], min_coverage=0)
    assert result["markdown"] == layer(words=3)


def test_the_reported_share_is_floored_so_it_never_reaches_the_threshold(monkeypatch, digital_pdf):
    tokens = [a + b for a in "abcdefghij" for b in "klmnopqrstuvw"][:125]
    result = text_lane_over(monkeypatch, digital_pdf, [" ".join(tokens)], [" ".join(tokens[13:])], min_coverage=0.9)
    assert result["reason"] == "the text lane lost part of the text layer: page 1 kept 89%"


def ocr_lane_over(monkeypatch, healthy, text_pages: dict, ocr_pages: dict, layers: dict, calls=None) -> dict:
    monkeypatch.setattr(conversion, "_convert", scripted_convert(text_pages, ocr_pages, calls))
    text_result = {"markdown": conversion.join_pages(text_pages), "page_markdown": text_pages, "pages": len(text_pages)}
    return conversion.ocr_lane("doc.pdf", True, {"layers": layers, "text_result": text_result})


def test_ocr_of_the_lossy_pages_is_spliced_into_the_text_lanes_result_in_page_order(monkeypatch, healthy):
    calls = []
    text_pages = {0: "Intact 1,5 mg", 1: "lost", 2: "Intact 2.5 mg", 3: "lost"}
    ocr_pages = {1: layer(), 3: layer()}
    result = ocr_lane_over(monkeypatch, healthy, text_pages, ocr_pages, {1: layer(), 3: layer()}, calls)
    assert calls == [{"disable_ocr": False, "force_ocr": True, "page_range": [1, 3]}]
    assert result["markdown"] == f"Intact 1,5 mg\n\n{layer()}\n\nIntact 2.5 mg\n\n{layer()}"
    assert (result["pages"], result["ocr_pages"]) == (4, 2)


def test_a_one_page_document_takes_its_ocr_page(monkeypatch, healthy):
    result = ocr_lane_over(monkeypatch, healthy, {0: layer(words=4)}, {0: layer()}, {0: layer()})
    assert (result["markdown"], result["pages"], result["ocr_pages"]) == (layer(), 1, 1)


def test_a_document_where_every_page_is_lossy_is_all_ocr(monkeypatch, healthy):
    lost = {0: layer(words=3), 1: layer(words=4)}
    result = ocr_lane_over(monkeypatch, healthy, lost, {0: layer(), 1: layer()}, {0: layer(), 1: layer()})
    assert result["markdown"] == f"{layer()}\n\n{layer()}" and result["ocr_pages"] == 2


def test_an_ocr_page_that_kept_less_than_the_text_lanes_page_is_not_used(monkeypatch, healthy):
    text_pages = {0: layer(words=6), 1: layer(words=6)}
    ocr_pages = {0: layer(words=3), 1: layer()}
    result = ocr_lane_over(monkeypatch, healthy, text_pages, ocr_pages, {0: layer(), 1: layer()})
    assert result["markdown"] == f"{layer(words=6)}\n\n{layer()}" and result["ocr_pages"] == 1


def test_an_empty_ocr_page_keeps_the_text_lanes_page_without_failing_the_job(monkeypatch, healthy):
    text_pages = {0: layer(words=4), 1: layer(words=4)}
    result = ocr_lane_over(monkeypatch, healthy, text_pages, {0: "", 1: ""}, {0: layer(), 1: layer()})
    assert result["markdown"] == f"{layer(words=4)}\n\n{layer(words=4)}" and result["ocr_pages"] == 0


def test_digitless_ocr_pages_do_not_fail_a_partial_conversion_that_requires_digits(monkeypatch, healthy):
    monkeypatch.setattr(conversion, "_limits", TriageLimits(require_digits=True))
    result = ocr_lane_over(monkeypatch, healthy, {0: layer(words=4)}, {0: layer()}, {0: layer()})
    assert result["markdown"] == layer() and result["ocr_pages"] == 1


def test_an_ocr_page_that_misreads_a_number_is_not_used(monkeypatch, healthy):
    layer_text = f"{layer(words=6)} 5,5 3.7 4444"
    text_pages = {0: f"{layer(words=4)} 5,5 3.7 4444"}
    for misread in (f"{layer()} 5,8 3.7 4444", f"{layer()} 5.5 3.7 4444"):
        result = ocr_lane_over(monkeypatch, healthy, text_pages, {0: misread}, {0: layer_text})
        assert result["markdown"] == text_pages[0] and result["ocr_pages"] == 0


def test_an_ocr_page_that_only_dropped_a_line_is_used(monkeypatch, healthy):
    layer_text = f"{layer(words=6)} 5,5 3.7 4444"
    text_pages = {0: f"{layer(words=4)} 5,5 3.7 4444"}
    result = ocr_lane_over(monkeypatch, healthy, text_pages, {0: f"{layer()} 5,5 3.7"}, {0: layer_text})
    assert result["markdown"] == f"{layer()} 5,5 3.7" and result["ocr_pages"] == 1


def test_a_dead_ocr_server_fails_a_partial_conversion_too(monkeypatch, healthy):
    healthy["healthy"] = False
    with pytest.raises(OcrServerUnavailable):
        ocr_lane_over(monkeypatch, healthy, {0: "lost"}, {0: layer()}, {0: layer()})


def test_the_marker_config_selects_the_requested_pages():
    assert conversion.marker_config(disable_ocr=False, force_ocr=True, page_range=[1, 3])["page_range"] == [1, 3]


def test_the_marker_config_stops_ignoretext_from_dropping_repeated_blocks():
    config = conversion.marker_config(disable_ocr=True, force_ocr=False)
    assert config["IgnoreTextProcessor_common_element_min_blocks"] > 1000


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
