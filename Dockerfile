# syntax=docker/dockerfile:1
# 默认镜像只安装轻量依赖；SQLite 索引由 volume 提供。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY agent ./agent
COPY shared ./shared
COPY online_retrieval ./online_retrieval
COPY offline_ingestion ./offline_ingestion
COPY legal_mcp ./legal_mcp
COPY web ./web
COPY data ./data
COPY run_api.py run_mcp.py ./

EXPOSE 8000

# Web 页面和 API 在同一进程提供。
CMD ["python", "-m", "uvicorn", "online_retrieval.api:app", "--host", "0.0.0.0", "--port", "8000"]
