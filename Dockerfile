FROM rocm/pytorch:latest

# Set environment variables for better compatibility
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies (if any are needed by ONNX/Brevitas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the project and optional dependencies from pyproject.toml
WORKDIR /workspace/quickdraw-brevitas
COPY . /workspace/quickdraw-brevitas
RUN pip install --no-cache-dir ".[full]"

# Set default command
CMD ["bash"]
