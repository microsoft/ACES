#!/bin/bash
# Build script for Excytin Demo Domain Docker images

set -e

echo "🔨 Building Excytin Demo Domain Images"
echo "======================================"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Build SABER server for excytin domain
echo "📦 Building SABER server for excytin domain..."
# Build from repo root with the server Dockerfile
CURRENT_DIR=$(pwd)
cd ../../../
docker build -f domains/excytin_demo/docker/Dockerfile.server -t saber-excytin-server:latest .
cd "$CURRENT_DIR"

# Build Excytin Demo client
echo "📦 Building Excytin Demo client..."
cd ../../../
docker build -f domains/excytin_demo/docker/Dockerfile.client -t saber-excytin-client:latest domains/excytin_demo/
cd "$CURRENT_DIR"

# Build Excytin Demo sandbox
echo "📦 Building Excytin Demo sandbox..."
cd ../../../
docker build -f domains/excytin_demo/docker/Dockerfile.sandbox -t saber-excytin-sandbox:latest domains/excytin_demo/
cd "$CURRENT_DIR"

# Build custom MySQL image with SQL files
echo "📦 Building custom MySQL image with SQL data..."
cd ../../../
docker build -f domains/excytin_demo/docker/db/Dockerfile.incident_5 -t saber-excytin-incident-5:latest domains/excytin_demo/
cd "$CURRENT_DIR"

# Verify images were built
echo "✅ Verifying built images..."
docker images | grep 'saber-excytin'

echo ""
echo "🎉 SABER Excytin Demo images built successfully!"
echo ""
echo "Available images:"
echo "  • saber-excytin-server:latest  - SABER server for excytin domain"
echo "  • saber-excytin-client:latest  - Demo client for testing enhanced logging"
echo "  • saber-excytin-sandbox:latest  - Sandbox execution environment with mysql client"
echo "  • saber-excytin-incident-5:latest   - Custom MySQL with SQL data (Docker-in-Docker workaround)"
echo ""
echo "Next steps:"
echo "  • Run: docker-compose up -d"
echo "  • Test: docker exec -it saber-excytin-client uv run demo_client.py"
