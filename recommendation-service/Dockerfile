FROM python:3.11-slim

RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

COPY app ./app

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.getenv('SERVER_PORT','8081')+'/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${SERVER_HOST:-0.0.0.0} --port ${SERVER_PORT:-8081}"]