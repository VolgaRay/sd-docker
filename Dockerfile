FROM nvcr.io/nvidia/pytorch:24.05-py3

# Install basic OS deps
RUN apt-get update && apt-get install -y vim wget git curl

# Upgrade pip and install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Create workspace directory
RUN mkdir -p /workspace/sd
WORKDIR /workspace/sd
COPY . /workspace/sd

ENV HF_HOME=/root/.cache/huggingface
