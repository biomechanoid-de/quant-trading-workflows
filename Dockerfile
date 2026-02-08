# Dockerfile for Quant Trading Workflows on ARM64 (Raspberry Pi)

FROM python:3.11-slim-bookworm

# Labels for Container Registry
LABEL org.opencontainers.image.source="https://github.com/biomechanoid-de/quant-trading-workflows"
LABEL org.opencontainers.image.description="Quant Trading Workflows for Pi Cluster"
LABEL org.opencontainers.image.licenses="MIT"

# Set PYTHONPATH for imports to work
ENV PYTHONPATH=/root
ENV PYTHONUNBUFFERED=1
ENV FLYTE_INTERNAL_IMAGE=""

WORKDIR /root

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . /root

# Install uv for fast dependency resolution, then install project
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache .

# Clean up build dependencies
RUN apt-get purge -y --auto-remove build-essential gcc

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import flytekit; print('OK')" || exit 1

# Default command (overridden by Flyte)
CMD ["python", "-m", "flytekit.bin.entrypoint"]
