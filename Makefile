APP_NAME="ocr-extractor"

docker/build:
	docker-compose build ${APP_NAME}

docker/up:
	docker-compose up -d

docker/down:
	docker-compose down --remove-orphans

local/install:
	poetry install

local/run:
	poetry run python src/main.py