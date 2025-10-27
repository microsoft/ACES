# SABER Docker Architecture

This directory contains the base Docker images for the SABER system. These images provide common functionality that can be extended by domain-specific Dockerfiles.

## Base Images

### `saber/server:latest`
**Base SABER server image** with:
- Python 3.11-slim base
- Docker CLI for Docker-in-Docker execution
- SABER source code and dependencies
- Common server setup and configuration
- Exposed ports: 8000 (REST API), 8001 (MCP API)

### `saber/client:latest`
**Base SABER client image** with:
- Python 3.11-slim base
- uv package manager for Python dependencies
- Common client setup and configuration

### `saber/sandbox:latest`
**Base sandbox execution environment** with:
- Ubuntu 22.04 base
- Common security and networking tools
- Python 3 with pip
- Non-root user setup for security

### `saber/execution:latest`
**Base execution environment** with:
- Ubuntu 22.04 base
- Penetration testing and security tools
- Network utilities and analysis tools
- Development tools and Python packages

## Building Base Images

To build all base images:

```bash
cd /path/to/saber/docker
./build-base-images.sh
```

This creates the base images that domains can extend.

## Domain Extension Pattern

Domains should extend these base images with minimal Dockerfiles:

### Server Extension Example
```dockerfile
FROM saber/server:latest

# Copy domain-specific config and data
COPY server/config/ /app/config/
COPY server/data/ /app/data/

# Set domain-specific environment
ENV SABER_DOMAIN=my_domain

# Domain-specific setup if needed
RUN some-domain-specific-command

CMD ["python", "-m", "saber.server", "--start", "--domain", "my_domain"]
```

### Client Extension Example
```dockerfile
FROM saber/client:latest

# Copy domain client code
COPY client/ /app/client/

# Install domain-specific dependencies
WORKDIR /app/client
RUN uv sync

CMD ["tail", "-f", "/dev/null"]
```

### Sandbox Extension Example
```dockerfile
FROM saber/sandbox:latest

# Install domain-specific tools
USER root
RUN apt-get update && apt-get install -y domain-specific-tool
USER tooluser

# Copy domain-specific scripts
COPY scripts/ /workspace/scripts/
```

## Directory Structure

```
docker/
├── build-base-images.sh          # Builds all base images
├── Dockerfile.saber_server        # Base server image
├── Dockerfile.saber_client        # Base client image  
├── Dockerfile.saber_sandbox       # Base sandbox image
├── Dockerfile.saber_execution     # Base execution image
└── README.md                      # This file

domains/my_domain/
└── server/
    ├── config/                    # Task configs, prompts, etc.
    └── docker/                    # Domain Docker files
        ├── Dockerfile.server      # Extends saber/server
        ├── Dockerfile.client      # Extends saber/client
        └── Dockerfile.sandbox     # Extends saber/sandbox
```

## Benefits

1. **Consistency**: All domains use the same base setup
2. **Maintainability**: Common changes only need to be made in one place
3. **Efficiency**: Base images can be cached and reused
4. **Simplicity**: Domain Dockerfiles focus only on domain-specific concerns
5. **Standardization**: Clear patterns for extending base functionality

## Migration Guide

To migrate existing domain Dockerfiles:

1. Identify common setup shared with base images
2. Remove common setup from domain Dockerfiles
3. Change `FROM python:3.11-slim` to `FROM saber/server:latest` (or appropriate base)
4. Keep only domain-specific configuration and tools
5. Update build scripts to check for base images
6. Test builds to ensure functionality is preserved
