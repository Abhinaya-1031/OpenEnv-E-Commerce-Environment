FROM ghcr.io/astral-sh/uv:python3.11-slim

# Create application directory
WORKDIR /app

# Install dependencies first (caching layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY . .

# Final installation of the project itself
RUN uv pip install -e .

EXPOSE 8000

# Start server using the entry point defined in pyproject.toml
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
