# Food Recommendation Service

Standalone FastAPI microservice for food recommendations. The service reads from the current PostgreSQL database, exposes recommendation APIs, and keeps the recommendation logic isolated so the ranking engine can later be replaced with ML or AI models.

## Features

- FastAPI API surface for recommendation, health, and feedback
- Read-only PostgreSQL access with schema discovery for common food-platform table names
- Hybrid recommendation pipeline with popularity, content match, user-profile affinity, and context scoring
- Short-lived in-memory cache for repeated requests
- Docker-ready deployment and basic tests

## Endpoints

- `GET /health`
- `GET /v1/recommendations`
- `POST /v1/recommendations/query`
- `POST /v1/feedback`

## Local Run

```bash
pip install -e .[dev]
uvicorn app.main:app --host 192.168.30.128 --port 8081
```

## Environment

The service reads configuration from `.env`:

```env
SERVER_HOST=192.168.30.128
DATABASE_URL="postgresql://postgres:123456@192.168.30.128:5432/ai_food_db?schema=public"
SERVER_PORT=8081
```

## Recommendation Strategy

The default strategy is a lightweight hybrid ranker:

- popularity/trending fallback
- content-based similarity on category, cuisine, tags, and price
- user-profile affinity from past interactions when available
- context score based on meal time and request filters

When the current database schema does not expose expected interaction tables, the service falls back to popularity plus content/context signals instead of failing.