FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Force Hugging Face / Transformers to stay offline during evaluation.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN python3 -m pip install --upgrade pip

# Install repository dependencies when available.
RUN if [ -f requirements.txt ]; then \
      python3 -m pip install --no-cache-dir -r requirements.txt; \
    fi

# Install the repository itself when it is packaged as a Python project.
RUN if [ -f pyproject.toml ] || [ -f setup.py ]; then \
      python3 -m pip install --no-cache-dir -e .; \
    fi

ENV PYTHONPATH=/workspace

CMD ["bash"]
