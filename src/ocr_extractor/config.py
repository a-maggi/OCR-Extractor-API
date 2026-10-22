from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Shared secret the app sends as "Authorization: Bearer <key>". Required: the service refuses to start without it.
    OCR_API_KEY: str = Field(min_length=16)
    # Hosts a job URL may point to. "*.ufs.sh" matches any subdomain. Redirects are re-checked against this list.
    ALLOWED_SOURCE_HOSTS: str = "utfs.io,*.ufs.sh"

    # Parallel conversions per lane. Digital PDFs never touch the OCR model, so the text lane can be wider.
    TEXT_WORKERS: int = Field(default=2, ge=1)
    OCR_WORKERS: int = Field(default=1, ge=1)
    # torch/OpenMP threads per worker process, so workers don't fight over the same cores
    TORCH_THREADS_PER_WORKER: int = Field(default=2, ge=1)

    # Unfinished jobs (downloading, queued or converting) beyond this get a 429
    MAX_QUEUED_JOBS: int = Field(default=50, ge=1)
    # A conversion running longer than this is killed and its job fails
    JOB_TIMEOUT_SECONDS: float = Field(default=600, gt=0)
    # Finished jobs stay readable this long, so the app can pick a result up after a restart
    RESULT_TTL_SECONDS: float = Field(default=3600, gt=0)

    MAX_DOWNLOAD_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)
    DOWNLOAD_TIMEOUT_SECONDS: float = Field(default=30, gt=0)

    # A page with images and fewer non-whitespace characters than this counts as an image
    MIN_PAGE_CHARS: int = Field(default=20, ge=1)
    # A page whose images cover at least this share of it needs IMAGE_PAGE_MIN_CHARS of visible text to count as
    # digital; less is a scan with a digital stamp or footer on top
    IMAGE_COVERAGE_THRESHOLD: float = Field(default=0.65, gt=0, le=1)
    IMAGE_PAGE_MIN_CHARS: int = Field(default=200, ge=1)
    SSE_PING_SECONDS: float = Field(default=15, gt=0)

    # Start llama-server (the Surya OCR model) at boot instead of on the first scanned document
    PREWARM_VLM: bool = False
    # Run one small conversion at boot, so the layout and OCR-error servers are up before /readyz says so
    WARMUP: bool = True
    # First wait before retrying a failed startup; it doubles on each failure, up to 5 minutes
    STARTUP_RETRY_SECONDS: float = Field(default=10, gt=0)

    @property
    def allowed_hosts(self) -> list[str]:
        return [host.strip().lower() for host in self.ALLOWED_SOURCE_HOSTS.split(",") if host.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
