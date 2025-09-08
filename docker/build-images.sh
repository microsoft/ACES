#!/bin/bash
# Build script for SABER base Docker images

set -e

echo "🔨 Building SABER Base Images"
echo "============================="

# Get the directory of this script (should be /docker)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Build from repo root context
cd ..

echo "📦 Building SABER base server image..."
docker build -f docker/Dockerfile.saber_server -t saber/server:latest .

echo "📦 Building SABER base client image..."
docker build -f docker/Dockerfile.saber_client -t saber/client:latest .

echo "📦 Building SABER base sandbox image..."
docker build -f docker/Dockerfile.saber_sandbox -t saber/sandbox:latest .

echo "📦 Building SABER base execution environment..."
docker build -f docker/Dockerfile.saber_execution -t saber/execution:latest .

# Verify images were built
echo "✅ Verifying built base images..."
docker images | grep 'saber/'

echo ""
echo "🎉 SABER base images built successfully!"
echo ""
echo "Available base images:"
echo "  • saber/server:latest       - Base SABER server with Docker-in-Docker"
echo "  • saber/client:latest       - Base SABER client with uv package management"
echo "  • saber/sandbox:latest      - Base sandbox environment with common tools"
echo "  • saber/execution:latest    - Base execution environment with security tools"
echo ""
echo "💡 Use these as base images in domain-specific Dockerfiles"
