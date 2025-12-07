# ai_redteam Sandbox Image
# Provides execution environment for both red and blue team agents

FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies including MySQL client and supervisor
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    netcat-openbsd \
    mariadb-client \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including WebSocket libraries
RUN pip install --no-cache-dir \
    requests \
    mysql-connector-python \
    websockets==12.0 \
    aiohttp==3.9.0

# Copy WebSocket daemon scripts
COPY websocket_daemon.py /usr/local/bin/websocket_daemon.py
COPY inject_via_ipc.py /usr/local/bin/inject_via_ipc.py
RUN chmod +x /usr/local/bin/websocket_daemon.py /usr/local/bin/inject_via_ipc.py

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create workspace directory and log directories
RUN mkdir -p /workspace && chmod 777 /workspace && \
    mkdir -p /var/log/supervisor && chmod 777 /var/log/supervisor

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Start supervisor (manages daemon)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
