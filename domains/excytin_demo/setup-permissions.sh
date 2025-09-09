#!/bin/bash

# SABER Demo Docker Environment Setup
# Sets proper user permissions for log file access

echo "🚀 Setting up SABER excytin demo with proper user permissions..."

# Get current user and group IDs
export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)

echo "📋 Using UID:GID = $DOCKER_UID:$DOCKER_GID ($(whoami))"

# Fix existing log permissions
echo "🔧 Fixing existing log file permissions..."
sudo chown -R $DOCKER_UID:$DOCKER_GID ./client/logs/ || true
sudo chown -R $DOCKER_UID:$DOCKER_GID ./server/logs/ || true

# Restart containers with proper user mapping
echo "🐳 Restarting containers with user mapping..."
docker compose down
docker compose up -d

echo "✅ SABER demo is ready with proper log permissions!"
echo "📊 Log files will now be created with your user ownership"
echo ""
echo "To run evaluations:"
echo "  docker exec -it saber-excytin-client bash"
echo "  cd /app/client && uv run python example_saber_task.py"
echo ""
echo "To check logs:"
echo "  ls -la ./client/logs/  # Client evaluation logs"
echo "  ls -la ./server/logs/  # Server operation logs"
