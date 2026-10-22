from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

MAX_REDIRECTS = 3


class DownloadError(Exception):
    pass


def host_allowed(url: str, allowed_hosts: list[str]) -> bool:
    """True for https URLs whose host is in the allowlist. "*.example.com" matches subdomains only."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower()
    for pattern in allowed_hosts:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


async def download_pdf(
    url: str,
    dest: Path,
    *,
    allowed_hosts: list[str],
    max_bytes: int,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Streams a PDF to dest. Follows redirects by hand so every hop is checked against the allowlist.

    Errors name the host only: the full URL is enough to download the document, so it stays out of logs."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            host = urlsplit(url).hostname
            if not host_allowed(url, allowed_hosts):
                raise DownloadError(f"Host not allowed: {host}")

            try:
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    if response.status_code != 200:
                        raise DownloadError(f"Download from {host} failed with HTTP {response.status_code}")

                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        raise DownloadError(f"File is larger than {max_bytes} bytes")

                    size = 0
                    with dest.open("wb") as file:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise DownloadError(f"File is larger than {max_bytes} bytes")
                            file.write(chunk)
                    break
            except httpx.HTTPError as error:
                raise DownloadError(f"Download from {host} failed: {type(error).__name__}") from None
        else:
            raise DownloadError("Too many redirects")

    with dest.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise DownloadError("Not a PDF")
