import logging
from pathlib import Path

import httpx
import pytest

from ocr_extractor.config import get_settings
from ocr_extractor.download import DownloadError, download_pdf, host_allowed

ALLOWED = ["utfs.io", "*.ufs.sh"]
PDF = b"%PDF-1.4 test"


@pytest.mark.parametrize(
    "url, allowed",
    [
        ("https://utfs.io/f/abc", True),
        ("https://x8k2.ufs.sh/f/abc", True),
        ("http://utfs.io/f/abc", False),  # https only
        ("https://ufs.sh/f/abc", False),  # "*." matches subdomains, not the bare domain
        ("https://evilufs.sh/f/abc", False),
        ("https://utfs.io.evil.com/f/abc", False),
        ("https://169.254.169.254/latest/meta-data", False),
        ("file:///etc/passwd", False),
    ],
)
def test_host_allowed(url: str, allowed: bool):
    assert host_allowed(url, ALLOWED) is allowed


async def fetch(handler, url: str, dest: Path, max_bytes: int = 1024) -> None:
    await download_pdf(
        url, dest, allowed_hosts=ALLOWED, max_bytes=max_bytes, timeout=5, transport=httpx.MockTransport(handler)
    )


async def test_follows_redirects_within_the_allowlist(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "utfs.io":
            return httpx.Response(307, headers={"location": "https://app1.ufs.sh/f/abc"})
        return httpx.Response(200, content=PDF)

    dest = tmp_path / "doc.pdf"
    await fetch(handler, "https://utfs.io/f/abc", dest)
    assert dest.read_bytes() == PDF


async def test_refuses_a_redirect_to_another_host(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://ocr-internal:8000/v1/jobs"})

    with pytest.raises(DownloadError, match="Host not allowed"):
        await fetch(handler, "https://utfs.io/f/abc", tmp_path / "doc.pdf")


async def test_refuses_files_over_the_size_cap(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF + b"x" * 2048)

    with pytest.raises(DownloadError, match="larger than"):
        await fetch(handler, "https://utfs.io/f/abc", tmp_path / "doc.pdf")


async def test_refuses_files_that_are_not_pdfs(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>login</html>")

    with pytest.raises(DownloadError, match="Not a PDF"):
        await fetch(handler, "https://utfs.io/f/abc", tmp_path / "doc.pdf")


async def test_reports_http_errors(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(DownloadError, match="HTTP 404"):
        await fetch(handler, "https://utfs.io/f/abc", tmp_path / "doc.pdf")


async def test_stops_reading_a_body_with_no_content_length_at_the_size_cap(tmp_path: Path):
    pulled = 0

    async def body():
        nonlocal pulled
        yield PDF
        for _ in range(10_000):
            pulled += 1024
            yield b"x" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())  # An async body is sent chunked, without Content-Length

    with pytest.raises(DownloadError, match="larger than"):
        await fetch(handler, "https://utfs.io/f/abc", tmp_path / "doc.pdf", max_bytes=64 * 1024)
    assert pulled <= 64 * 1024 + 1024


async def test_errors_name_the_host_but_never_the_document_url(tmp_path: Path):
    """An UploadThing URL is enough to download the document, so it must stay out of errors and logs."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused for {request.url}", request=request)

    with pytest.raises(DownloadError) as raised:
        await fetch(handler, "https://abc123.ufs.sh/f/SECRETFILEKEY", tmp_path / "doc.pdf")
    assert str(raised.value) == "Download from abc123.ufs.sh failed: ConnectError"
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


def test_the_service_keeps_httpx_request_urls_out_of_its_logs(monkeypatch):
    monkeypatch.setenv("OCR_API_KEY", "test-key-0123456789")
    get_settings.cache_clear()
    try:
        import ocr_extractor.main  # noqa: F401  Configures logging on import

        assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
        assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
    finally:
        get_settings.cache_clear()
