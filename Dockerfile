FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group train --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --no-group train
COPY data/serving ./data/serving

RUN useradd --create-home appuser
USER appuser

ENV RECSYS_ARTIFACTS_DIR=/app/data/serving
EXPOSE 8000
HEALTHCHECK CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/app/.venv/bin/uvicorn", "recsys.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]