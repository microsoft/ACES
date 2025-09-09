#!/bin/bash

# SABER Demo Docker Environment Setup
# Sets proper user permissions for log file access

echo "🚀 Setting up SABER excytin demo with proper user permissions..."

# Get current user and group IDs
export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)
export DOCKER_GROUP_ID=$(getent group docker | cut -d: -f3)

echo "📋 Using UID:GID = $DOCKER_UID:$DOCKER_GID ($(whoami))"
echo "📋 Using Docker Group ID = $DOCKER_GROUP_ID"

# Fix existing log permissions
echo "🔧 Fixing existing log file permissions..."
sudo chown -R $DOCKER_UID:$DOCKER_GID ./client/logs/ || true
sudo chown -R $DOCKER_UID:$DOCKER_GID ./server/logs/ || true

# Restart containers with proper user mapping
echo "🐳 Restarting containers with user mapping..."
docker compose down
docker compose up -d

# Fix container app directory permissions for cache creation
echo "🔧 Fixing container permissions for cache directories..."
docker exec saber-excytin-client bash -c "sudo chown -R $DOCKER_UID:$DOCKER_GID /app && sudo chmod -R 755 /app" 2>/dev/null || \
docker exec --user root saber-excytin-client bash -c "chown -R $DOCKER_UID:$DOCKER_GID /app && chmod -R 755 /app" || \
echo "⚠️  Could not fix container permissions - you may need to run containers as root initially"

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
