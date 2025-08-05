#!/bin/bash
# Entrypoint script for SABER Basic Security Domain Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting SABER Network Investigation Domain Server${NC}"

# Change to workspace directory
cd /workspace
echo -e "${BLUE}📁 Working directory: $(pwd)${NC}"

# Environment info
echo -e "${YELLOW}🔧 Environment info...${NC}"
echo -e "   Python: $(python --version 2>&1 || echo 'Not found')"
echo -e "   UV: $(uv --version 2>&1 || echo 'Not found')"
echo -e "   Docker: $(docker --version 2>&1 || echo 'Not found')"

# Set up Python path
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
export PYTHONPATH="/workspace/src:$PYTHONPATH"

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
uv sync || {
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
}

# Verify Docker access
echo -e "${YELLOW}🐳 Checking Docker access...${NC}"
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker access verified${NC}"
else
    echo -e "${RED}❌ Docker access failed - container may need privileged mode${NC}"
    exit 1
fi

# Create data directories
echo -e "${YELLOW}📁 Creating data directories...${NC}"
mkdir -p /workspace/data/samples
mkdir -p /workspace/data/logs
mkdir -p /workspace/data/results

# Set default values
export SABER_DOMAIN=${SABER_DOMAIN:-network_investigation}
export SABER_PORT=${SABER_PORT:-8000}
export SABER_LOG_LEVEL=${SABER_LOG_LEVEL:-INFO}

echo -e "${YELLOW}⚙️ Configuration:${NC}"
echo -e "   Domain: $SABER_DOMAIN"
echo -e "   Port: $SABER_PORT"
echo -e "   Log Level: $SABER_LOG_LEVEL"

# Start the SABER server
echo -e "${GREEN}🎯 Starting SABER server on port $SABER_PORT...${NC}"
exec uv run python -m network_investigation_server \
    --domain "$SABER_DOMAIN" \
    --port "$SABER_PORT" \
    --log-level "$SABER_LOG_LEVEL"
