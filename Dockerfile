# syntax=docker/dockerfile:1
# Python 3.12 是当前依赖的保守运行版本；模型与 Chroma 索引通过 volume 提供，
# 不烘焙进镜像，从而避免数 GB 镜像和数据/代码版本耦合。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

# 单容器默认启动 API；compose.yaml 会为 MCP 服务覆盖此命令。
CMD ["python", "-m", "uvicorn", "online_retrieval.api:app", "--host", "0.0.0.0", "--port", "8000"]
