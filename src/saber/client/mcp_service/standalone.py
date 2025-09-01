#!/usr/bin/env python3
"""
Standalone MCP Sidecar Service Entry Point

Independent entry point for the MCP sidecar service that doesn't depend
on the full SABER client package infrastructure.
"""

import os
import sys
from pathlib import Path

# Add the source directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent  # Go up to src/
sys.path.insert(0, str(src_dir))

# Import only what we need for the MCP service
from saber.client.mcp_service.main import start_sidecar_service

if __name__ == "__main__":
    # Read environment for configurability inside the container
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8002"))
    saber_mcp_url = os.environ.get("SABER_MCP_URL", "http://localhost:8001")
    start_sidecar_service(host=host, port=port, saber_mcp_url=saber_mcp_url)
