docker/build:
	docker compose build ocr

docker/up:
	docker compose up -d

docker/down:
	docker compose down --remove-orphans

# Runs the image with port 8000 published, for trying it out locally (compose publishes nothing)
docker/dev:
	docker run --rm -p 8000:8000 --env-file .env ocr-extractor-ocr

local/install:
	uv sync

local/run:
	uv run uvicorn ocr_extractor.main:app --host 127.0.0.1 --port 8000 --workers 1

test:
	uv run pytest
