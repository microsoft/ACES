# AI Red Team Blue Agent Sandbox Image
# Simple execution environment for blue team agent with database access
# No WebSocket daemon needed - this is the target agent

FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies including MySQL client
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    netcat-openbsd \
    mariadb-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    mysql-connector-python

# Create workspace directory
RUN mkdir -p /workspace && chmod 777 /workspace

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Keep container running for command execution
CMD ["tail", "-f", "/dev/null"]
