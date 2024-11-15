# OCR Extractor API

A FastAPI-based service that provides OCR (Optical Character Recognition) extraction from PDFs using Marker. The service can process PDFs both locally and remotely.

## Features

- Extract text from PDFs using OCR
- Support for multiple languages
- Image extraction from PDFs
- Local and remote PDF processing
- Force OCR processing when needed
- Optional pagination support

## Technology Stack

- [Python 3.10+](https://www.python.org/) - **pre-requisite**
- [Poetry](https://python-poetry.org/) - **pre-requisite**
- [FastAPI](https://fastapi.tiangolo.com/)
- [Marker PDF](https://github.com/VikParuchuri/marker)
- [PyTorch](https://pytorch.org/)

## Installation and Running

### Using Docker (Recommended)

1. Build the Docker image:

```bash
make docker/build
```

2. Run the Docker container:

```bash
make docker/run
```

### Using Poetry

1. Install dependencies:

```bash
poetry install
```

2. Run the service:

```bash
poetry run python src/main.py
```

## API Documentation

### PDF Conversion Endpoint

**Endpoint:** `POST /marker`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| filepath | string | Yes | - | The path to the PDF file to convert |
| max_pages | integer | No | null | The maximum number of pages in the document to convert |
| langs | string | No | null | Languages to use for OCR, comma separated (e.g., "en,es"). Uses codes from [Surya's language file](https://github.com/VikParuchuri/surya/blob/master/surya/languages.py) |
| force_ocr | boolean | No | false | Force OCR on all pages. Warning: Can lead to worse results if PDFs already have good text |
| paginate | boolean | No | false | If true, separates output pages with horizontal rules containing page numbers |
| extract_images | boolean | No | true | Whether to extract images from the PDF |

#### Example Request

```bash
curl -X POST "http://localhost:8000/marker" -H "Content-Type: application/json" -d '{"filepath": "https://example.com/path/to/pdf.pdf", "max_pages": 10, "langs": "en,es", "force_ocr": false, "paginate": false, "extract_images": true}'
```


#### Response Format

The API returns a JSON object with the following structure:

```json
{
    "markdown": "Extracted text content in markdown format",
    "images": {
        "image_key": "base64_encoded_image_string"
    },
    "metadata": {
        "languages": ["detected_language_codes"],
        "filetype": "pdf",
        "pdf_toc": [],
        "pages": 5,
        "ocr_stats": {
            "ocr_pages": 0,
            "ocr_failed": 0,
            "ocr_success": 0,
            "ocr_engine": "none"
        },
        "block_stats": {
            "header_footer": 0,
            "code": 0,
            "table": 0,
            "equations": {
    "successful_ocr": 0,
    "unsuccessful_ocr": 0,
    "equations": 0
    }
    },
    "computed_toc": []
    },
    "success": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| markdown | string | The extracted text content in markdown format |
| images | object | Dictionary of extracted images (if any) as base64 encoded strings |
| metadata | object | Processing metadata and statistics |
| metadata.languages | array | Detected languages in the document |
| metadata.pages | integer | Total number of pages processed |
| metadata.ocr_stats | object | Statistics about OCR processing |
| metadata.block_stats | object | Statistics about different content blocks found |
| success | boolean | Whether the conversion was successful |

#### Error Response

In case of an error, the API returns:

```json
{
    "success": false,
    "error": "Error message description"
}
```

### Interactive Documentation

For interactive API documentation, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`