"""
SOLUTION — Exercise 1: Dockerize the Agent API
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import stat

OUTPUT_DIR = os.path.dirname(__file__)


def generate_dockerfile() -> str:
    return '''\
# ── Stage 1: Dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN useradd --create-home --shell /bin/bash agent
WORKDIR /app
RUN chown agent:agent /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=agent:agent . .

USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "phase4_production.week7_api_serving.exercises.ex1_fastapi_agent:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''


def generate_compose() -> str:
    return '''\
version: "3.9"

services:
  agent-api:
    build:
      context: ../../..
      dockerfile: phase6_capstone/week12_deployment/exercises/solutions/Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - ../../../.env
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://agent:secret@postgres:5432/agentdb
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  celery-worker:
    build:
      context: ../../..
      dockerfile: phase6_capstone/week12_deployment/exercises/solutions/Dockerfile
    command: celery -A phase4_production.week7_api_serving.exercises.ex3_celery_worker worker --loglevel=info --concurrency=4
    env_file:
      - ../../../.env
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agentdb
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
'''


def generate_dockerignore() -> str:
    return '''\
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/
*.egg-info/
dist/
build/
.eggs/
.venv/
venv/
env/
.env
.env.*
*.pem
*.key
.git/
.gitignore
.vscode/
.idea/
*.ipynb
*.ipynb_checkpoints/
htmlcov/
.coverage
coverage.xml
docs/
*.db
*.sqlite
*.csv
node_modules/
'''


def generate_entrypoint() -> str:
    return '''\
#!/bin/bash
# entrypoint.sh — startup script with pre-flight checks
set -e

echo "=== Agent API Startup ==="
echo "Model: ${MODEL:-not set}"
echo "Python: $(python --version)"

if [ -z "$MODEL" ]; then
  echo "ERROR: MODEL environment variable is not set"
  exit 1
fi

if [ -n "$REDIS_URL" ]; then
  echo "Waiting for Redis..."
  until python -c "import redis; r = redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; do
    sleep 2
  done
  echo "Redis is ready."
fi

exec uvicorn phase4_production.week7_api_serving.exercises.ex1_fastapi_agent:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers "${WORKERS:-2}" \
  --log-level "${LOG_LEVEL:-info}"
'''


FILES = {
    "Dockerfile": generate_dockerfile,
    "docker-compose.yml": generate_compose,
    ".dockerignore": generate_dockerignore,
    "entrypoint.sh": generate_entrypoint,
}


if __name__ == "__main__":
    generated = []
    for filename, gen_fn in FILES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            f.write(gen_fn())
        generated.append(path)
        print(f"  ✓ Generated: {filename}")

    # Make entrypoint executable
    ep_path = os.path.join(OUTPUT_DIR, "entrypoint.sh")
    os.chmod(ep_path, os.stat(ep_path).st_mode | stat.S_IEXEC)

    print(f"\n✅ Generated {len(generated)} Docker files in:\n  {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. python sol1_dockerfile.py  ← generates Dockerfile, docker-compose.yml etc.")
    print("  2. docker build -t agent-api .")
    print("  3. docker run --env-file .env -p 8000:8000 agent-api")
    print("  4. docker-compose up  ← full stack: agent + redis + postgres + celery")
