FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

RUN groupadd --gid 10001 raguser && \
    useradd --uid 10001 --gid raguser --create-home --shell /usr/sbin/nologin raguser

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
COPY data ./data
COPY artifacts ./artifacts

RUN pip install --no-cache-dir . && \
    chown -R raguser:raguser /app

USER raguser
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "hybrid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
