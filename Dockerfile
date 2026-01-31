FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-pip \
    python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1001 user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

USER user
WORKDIR $HOME/

RUN git clone https://github.com/quickmt/quickmt.git

# Install the package and dependencies
# This also installs the quickmt cli scripts
RUN pip install --break-system-packages --no-cache-dir ./quickmt

# Expose the default FastAPI port
EXPOSE 7860

# Hf Spaces expect the app on port 7860 usually
# We override the port via env var or CLI arg
CMD ["uvicorn", "quickmt.rest_server:app", "--host", "0.0.0.0", "--port", "7860"]
