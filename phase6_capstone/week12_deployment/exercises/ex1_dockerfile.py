"""
Exercise 1: Dockerize the Agent API
Goal: Generate production-ready Docker configuration for your FastAPI agent.

This script generates:
  - Dockerfile         — multi-stage build for small image size
  - docker-compose.yml — full stack: agent + redis + postgres
  - .dockerignore      — exclude unnecessary files
  - entrypoint.sh      — startup script with health check

Run:
  python ex1_dockerfile.py         # generate files
  docker build -t agent-api .      # build the image
  docker-compose up                # start the full stack
  curl http://localhost:8000/health # test

Tasks:
  1. Review the generated Dockerfile and understand each layer.
  2. Complete the TODO in generate_dockerfile() — add the HEALTHCHECK instruction.
  3. Complete the TODO in generate_compose() — add a postgres service.
  4. Build the image and run: docker run --env-file .env -p 8000:8000 agent-api
  5. (Bonus) Add a second Dockerfile stage for running tests inside Docker.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

OUTPUT_DIR = os.path.dirname(__file__)


# ── Dockerfile ────────────────────────────────────────────────────────────────

def generate_dockerfile() -> str:
    """
    Multi-stage Dockerfile:
      Stage 1 (builder): Install deps into a venv
      Stage 2 (runtime): Copy only the venv, run as non-root

    TODO: Add a HEALTHCHECK instruction before the CMD line:
      HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1
    """
    return '''\
# ── Stage 1: Dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash agent
WORKDIR /app
RUN chown agent:agent /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Expose the port FastAPI listens on
EXPOSE 8000

# TODO: Add HEALTHCHECK instruction here
# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
#   CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "phase4_production.week7_api_serving.exercises.ex1_fastapi_agent:app", \\
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''


# ── docker-compose.yml ────────────────────────────────────────────────────────

def generate_compose() -> str:
    """
    docker-compose.yml for local development with:
      - agent-api service
      - redis (for task queuing / Celery)
      - TODO: add postgres service

    TODO: Add a postgres service:
      postgres:
        image: postgres:16-alpine
        environment:
          POSTGRES_DB: agentdb
          POSTGRES_USER: agent
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
        volumes:
          - postgres_data:/var/lib/postgresql/data
        healthcheck:
          test: ["CMD-SHELL", "pg_isready -U agent"]
          interval: 10s
          timeout: 5s
          retries: 5
    """
    return '''\
version: "3.9"

services:
  # ── Agent API ──────────────────────────────────────────────────────────────
  agent-api:
    build:
      context: ../../..   # repo root (where Dockerfile lives)
      dockerfile: phase6_capstone/week12_deployment/exercises/Dockerfile
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
      # TODO: add postgres dependency here
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    restart: unless-stopped

  # ── Redis ──────────────────────────────────────────────────────────────────
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

  # ── Celery Worker ──────────────────────────────────────────────────────────
  celery-worker:
    build:
      context: ../../..
      dockerfile: phase6_capstone/week12_deployment/exercises/Dockerfile
    command: celery -A phase4_production.week7_api_serving.exercises.ex3_celery_worker worker --loglevel=info --concurrency=4
    env_file:
      - ../../../.env
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  # TODO: Add postgres service here

volumes:
  redis_data:
  # TODO: add postgres_data volume
'''


# ── .dockerignore ─────────────────────────────────────────────────────────────

def generate_dockerignore() -> str:
    return '''\
# Python
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# Secrets (NEVER include in Docker image)
.env
.env.*
*.pem
*.key

# Version control
.git/
.gitignore

# Development tools
.vscode/
.idea/
*.ipynb
*.ipynb_checkpoints/

# Test artifacts
htmlcov/
.coverage
coverage.xml

# Documentation
docs/
*.md

# Data files (add back what you need)
*.db
*.sqlite
*.csv
*.json

# Node
node_modules/
'''


# ── entrypoint.sh ─────────────────────────────────────────────────────────────

def generate_entrypoint() -> str:
    return '''\
#!/bin/bash
# entrypoint.sh — startup script with pre-flight checks

set -e  # Exit on any error

echo "=== Agent API Startup ==="
echo "Model: ${MODEL:-not set}"
echo "Python: $(python --version)"

# Check required env vars
if [ -z "$MODEL" ]; then
  echo "ERROR: MODEL environment variable is not set"
  exit 1
fi

# Wait for dependencies (Redis)
if [ -n "$REDIS_URL" ]; then
  echo "Waiting for Redis..."
  until python -c "import redis; r = redis.from_url('$REDIS_URL'); r.ping()" 2>/dev/null; do
    sleep 2
  done
  echo "Redis is ready."
fi

# Start the application
exec uvicorn phase4_production.week7_api_serving.exercises.ex1_fastapi_agent:app \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --workers "${WORKERS:-2}" \\
  --log-level "${LOG_LEVEL:-info}"
'''


# ── Main: Generate All Files ──────────────────────────────────────────────────

FILES = {
    "Dockerfile": (generate_dockerfile, "Dockerfile"),
    "docker-compose.yml": (generate_compose, "docker-compose.yml"),
    ".dockerignore": (generate_dockerignore, ".dockerignore"),
    "entrypoint.sh": (generate_entrypoint, "entrypoint.sh"),
}

if __name__ == "__main__":
    # Write to the exercises directory (sibling of this script)
    generated = []
    for label, (gen_fn, filename) in FILES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        content = gen_fn()
        with open(path, "w") as f:
            f.write(content)
        generated.append(path)
        print(f"  ✓ Generated: {filename}")

    # Make entrypoint executable
    import stat
    ep_path = os.path.join(OUTPUT_DIR, "entrypoint.sh")
    os.chmod(ep_path, os.stat(ep_path).st_mode | stat.S_IEXEC)

    print(f"\n✅ Generated {len(generated)} Docker files in:\n  {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Copy Dockerfile to repo root (or adjust build context in docker-compose.yml)")
    print("  2. docker build -t agent-api .")
    print("  3. docker run --env-file .env -p 8000:8000 agent-api")
    print("  4. docker-compose up  # full stack with Redis")
    print("\nTODO items:")
    print("  - Add HEALTHCHECK to Dockerfile")
    print("  - Add postgres service to docker-compose.yml")
    print("  - Complete the postgres dependency in agent-api service")
