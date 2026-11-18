FROM python:3.11-slim

RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir . \
 && pip uninstall -y setuptools wheel \
 && { find / -depth \( -iname 'setuptools*' -o -iname 'pkg_resources' -o -iname 'wheel-*' \) -path '*-packages/*' -exec rm -rf {} + 2>/dev/null || true; } \
 && rm -f /usr/local/lib/python3*/ensurepip/_bundled/*.whl

COPY app ./app

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.getenv('SERVER_PORT','8081')+'/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${SERVER_HOST:-0.0.0.0} --port ${SERVER_PORT:-8081}"]