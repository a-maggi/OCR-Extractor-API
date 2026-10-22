# OCR Extractor API

An internal service that turns PDFs into markdown with [Marker 2.0](https://github.com/datalab-to/marker), on CPU. It's meant to run next to the services that call it, on a private Docker network, never exposed publicly.

## How it works

- **Jobs, not requests.** A client submits a job (a PDF URL) and follows it over Server-Sent Events. Conversions never run inside the API process, so a slow one can't block the others.
- **Text first, OCR when needed.** Every job starts in the **text lane**, which checks the PDF's own text layer:
  1. Any of these sends the job to the OCR lane:
     - a page with no visible text (a scan);
     - a page whose text is mostly a scanner's invisible OCR layer;
     - a page mostly covered by images with little visible text (a scan with a digital stamp or footer on top).
  2. Marker's OCR-error model checks the text of the remaining pages; garbled text (a broken font encoding) sends the job to the OCR lane.
  3. Otherwise Marker converts the text layer in `fast` mode with OCR off: pure CPU, no OCR model.
  4. If that output is nearly empty (or, with `REQUIRE_DIGITS=true`, has no numbers at all), the job moves to the OCR lane once, with every page OCR'd.
- **The OCR lane** runs Marker in `fast` mode with OCR on, through the Surya OCR model served by `llama-server`. It keeps the good pages' text and OCRs only the scanned or garbled ones. A document with a scanner's OCR layer gets every page OCR'd, since Marker would otherwise keep that layer as if it were real text.
- **OCR results are checked.** Surya turns a failed OCR request into empty text, so a "finished" conversion could be missing pages. After each OCR-lane conversion the worker probes `llama-server`. If it's down, or every OCR'd page came back empty, the worker drops its connection and the job is retried once, which reattaches to or restarts the server. Output with no readable text fails the job.
- **Each lane has its own worker processes** (`TEXT_WORKERS`, `OCR_WORKERS`):
  - A conversion that runs past `JOB_TIMEOUT_SECONDS` is killed and its worker restarted, without touching the others.
  - A worker that dies while idle (the OOM killer, say) is replaced before its next job.
  - Starting a worker retries with backoff.
- **Jobs live in memory.** Clients keep the source of truth: after a restart here, unknown jobs return 404, and the client decides whether to resubmit or give up.

## API

All `/v1` routes need `Authorization: Bearer <OCR_API_KEY>`.

| Method and path | What it does |
|---|---|
| `POST /v1/jobs` | Body `{"job_id", "url", "force_ocr"}`. 202 with the job; 200 if that `job_id` already exists (same URL); 409 if it exists for another URL; 400 if the URL's host isn't allowed; 429 when `MAX_QUEUED_JOBS` jobs are unfinished; 503 (with `Retry-After`) while the workers are starting or startup keeps failing. `force_ocr` skips the text-layer check and OCRs every page. |
| `GET /v1/jobs/{job_id}` | The job, with `markdown` once it's done. 404 if unknown or expired. |
| `DELETE /v1/jobs/{job_id}` | Cancels a job: drops it from its queue, or kills the worker converting it. It ends as `failed` with `error: "Cancelled"`. A finished job is left as it is. 404 if unknown. |
| `GET /v1/jobs/{job_id}/events` | SSE. Sends the current state first, then `queued` (with `position`), `routed` (moved to the OCR lane, with `reason`), `converting`, and finally `done` (with `markdown`) or `failed` (with `error`). A `: ping` comment every 15s keeps the connection alive. |
| `GET /healthz` | The process is up. |
| `GET /readyz` | Workers are started and the warmup conversion finished. 503 with the error otherwise. |

A job looks like this:

```json
{
  "job_id": "3f0c…",
  "status": "done",
  "lane": "text",
  "position": null,
  "reason": null,
  "pages": 3,
  "ocr_pages": 0,
  "error": null,
  "duration_seconds": 4.2,
  "markdown": "| Item | Qty | Price | …"
}
```

## Configuration

| Variable | Default | What it controls |
|---|---|---|
| `OCR_API_KEY` | required | Shared secret clients send as a Bearer token, at least 16 characters |
| `ALLOWED_SOURCE_HOSTS` | required | Comma-separated hosts a job URL (and every redirect) may point to, e.g. `files.example.com,*.cdn.example.net`. `*.` matches subdomains |
| `REQUIRE_DIGITS` | `false` | Treat output without a single digit as unreadable. Turn it on when every real document has numbers (reports, invoices): number-free text-lane output then moves to the OCR lane, and number-free OCR output fails the job |
| `TEXT_WORKERS` | `2` | Digital PDFs converted at once |
| `OCR_WORKERS` | `1` | Scanned PDFs converted at once |
| `TORCH_THREADS_PER_WORKER` | `2` | torch/OpenMP threads per worker process |
| `MAX_QUEUED_JOBS` | `50` | Unfinished jobs before new ones get a 429 |
| `JOB_TIMEOUT_SECONDS` | `600` | A conversion running longer is killed |
| `RESULT_TTL_SECONDS` | `3600` | How long finished jobs stay readable |
| `MAX_DOWNLOAD_BYTES` / `DOWNLOAD_TIMEOUT_SECONDS` | `10485760` / `30` | Download limits |
| `MIN_PAGE_CHARS` | `20` | Fewer characters than this on a page with images counts as "mostly an image" |
| `IMAGE_COVERAGE_THRESHOLD` / `IMAGE_PAGE_MIN_CHARS` | `0.65` / `200` | A page whose images cover at least this share of it needs this many visible characters to count as digital; otherwise it's a scan with a stamp on top |
| `PREWARM_VLM` | `false` | Start `llama-server` at boot instead of on the first scanned document. If it fails, startup carries on and the first scanned document tries again |
| `WARMUP` | `true` | Run one small conversion at boot so `/readyz` means the layout and OCR-error servers are up |
| `STARTUP_RETRY_SECONDS` | `10` | Wait before retrying a failed startup; doubles each time, up to 5 minutes |
| `SURYA_INFERENCE_PARALLEL` | `2` (compose) | Concurrent OCR requests in `llama-server`; each reserves ~12k tokens of KV cache |
| `LLAMA_CPP_EXTRA_ARGS` | `--threads 2` (compose) | Extra `llama-server` flags, such as its CPU threads |
| `FAST_LAYOUT_NUM_THREADS` | `2` (compose) | Threads of the shared layout model server |

Rule of thumb: `TEXT_WORKERS × TORCH_THREADS_PER_WORKER` plus the `llama-server` threads should stay within the container's CPU limit (`OCR_CPUS`, default 4).

### Sizing memory

`OCR_MEMORY` defaults to `6g`. The container holds:
- the API process;
- one process per worker (`TEXT_WORKERS + OCR_WORKERS`);
- the shared layout and OCR-error servers;
- `llama-server`: ~1.5 GB of model plus ~12k tokens of KV cache per `SURYA_INFERENCE_PARALLEL` slot.

Measure it on your machine rather than trusting the default:

1. Run a scanned document through, so `llama-server` is up, and a few digital ones at once.
2. `docker stats <container>` shows the container's total against its limit.
3. `docker exec <container> ps -eo rss,args --sort=-rss` shows each process's resident memory, in KB.

Size `OCR_MEMORY` to the peak total plus ~25%. To fit a smaller box, lower `TEXT_WORKERS` first (one worker process each), then `SURYA_INFERENCE_PARALLEL`. If the OOM killer stops a worker, its job fails and the worker is replaced; `docker inspect <container>` shows `OOMKilled` if the whole container hit the limit.

Models download on first use into two volumes (`hf-cache`, `datalab-models`): the small layout and OCR-error models at boot, and the ~1.5 GB OCR model with the first scanned document (or at boot with `PREWARM_VLM=true`).

## Running it

```bash
cp .env.example .env   # set OCR_API_KEY and ALLOWED_SOURCE_HOSTS
make docker/build
make docker/up         # compose publishes no ports
make docker/dev        # or: run the image with port 8000 published, to try it locally
```

Without Docker (Python 3.12 and [uv](https://docs.astral.sh/uv/)): `make local/install`, then `make local/run`. OCR on a scanned document also needs `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) on the `PATH`.

`llama-server` comes from the image pinned in the Dockerfile's `LLAMA_CPP_IMAGE` (`server-b11669`, the v0.7.0 release). To bump it, set that line to the `server-bNNNN` tag built from the new release's commit. Release names (`vX.Y.Z`) aren't image tags, so find the build tag on the commit the release points to, check it's listed under the [package's tags](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp), and rebuild; the build checks `llama-server --version`.

Tests: `make test`.

## Deploying on Coolify

1. Create a **Docker Compose** resource from this repository. Leave it without a domain: it must not be public.
2. Set `OCR_API_KEY`, `ALLOWED_SOURCE_HOSTS` (and any setting above) in its environment variables.
3. Under **Advanced**, turn on **Connect To Predefined Network**, so the service joins the `coolify` network your other resources are on.
4. Deploy, then find the name clients should use for it. Coolify may add a suffix to the service name:
   ```bash
   docker network inspect coolify --format '{{range .Containers}}{{.Name}}{{"\n"}}{{end}}'
   ```
5. Point the client at `http://<that name>:8000` with the same `OCR_API_KEY`. Check it from the client's container: `wget -qO- http://<that name>:8000/healthz`.

## License

Code: Apache 2.0, like Marker. Marker's model weights use a modified AI Pubs Open Rail-M license: free for research, personal use, and companies under $5M in funding and revenue.
