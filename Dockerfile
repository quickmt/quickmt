FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-pip \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container
COPY . /app

# Create a non-root user for security
RUN useradd -m -u 1001 user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Install the package and dependencies
# This also installs the quickmt cli scripts
RUN pip install --break-system-packages --no-cache-dir /app/

# Expose the default FastAPI port
EXPOSE 7860

USER user

# Hf Spaces expect the app on port 7860 usually
# We override the port via env var or CLI arg
CMD ["uvicorn", "quickmt.rest_server:app", "--host", "0.0.0.0", "--port", "7860"]
