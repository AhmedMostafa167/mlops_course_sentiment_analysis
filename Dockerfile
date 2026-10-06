FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY config/ config/

# Install dependencies (CPU-only torch)
RUN uv sync --frozen --no-dev

# Expose the FastAPI port
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mlops_practitioner_course.api:app", "--host", "0.0.0.0", "--port", "8000"]
