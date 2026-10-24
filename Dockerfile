# Syntax: docker build -t uav-gnn-allocation .
# Run:    docker run --rm -v $(pwd)/results:/app/results uav-gnn-allocation

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for scientific computing, image processing & video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first for layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install imageio imageio-ffmpeg

# Copy project source and configuration
COPY src/ ./src/
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY README.md ./

# Install package in editable/standard mode
RUN pip install -e . --no-deps

# Create results volume directory
RUN mkdir -p /app/results

# Default entrypoint runs the 3D isometric simulation video generator
ENTRYPOINT ["python", "scripts/generate_gcbf_style_video.py"]
CMD ["--uavs", "20", "--tasks", "50", "--fps", "20", "--save-video", "results/uav_simulation_3d_isometric.mp4"]
