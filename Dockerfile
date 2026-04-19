FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv
COPY pyproject.toml .
COPY kinema/ kinema/
COPY recipes/ recipes/
RUN uv pip install --system .

WORKDIR /work

ENTRYPOINT ["python", "-m", "kinema"]
