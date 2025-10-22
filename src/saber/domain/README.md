# SABER Domain CLI

The SABER Domain CLI provides automated orchestration for SABER security evaluation domains with **zero manual configuration** - no more editing `.env` files!

## Overview

The domain CLI automatically handles:
- ✅ **Environment generation** - All Docker Compose variables generated automatically
- ✅ **Domain validation** - Manifest schema validation and structure checks
- ✅ **Port management** - Automatic port conflict detection
- ✅ **Image building** - Integrated Docker image building with proper tagging
- ✅ **Service orchestration** - Full Docker Compose lifecycle management
- ✅ **Resource packaging** - Compose files and schemas bundled with installation

## Installation

The domain CLI is included with SABER:

```bash
# Install SABER (includes domain CLI)
pip install saber

# Or for development
git clone <saber-repo>
cd saber
uv install -e .
```

## Quick Start

```bash
# List available domains
saber-domain list

# Start a domain (server + client) - everything configured automatically!
saber-domain start cybench

# Start with custom ports and rebuild images
saber-domain start cybench --rebuild --rest-port 9000 --mcp-port 9001

# Start with privileged access (if domain requires it)
saber-domain start cybench --profiles server,client,docker-socket

# Check domain status
saber-domain status

# Stop domain
saber-domain stop cybench
```

## Commands

### `list` - List Available Domains

```bash
# Basic listing
saber-domain list

# Detailed information with descriptions
saber-domain list --verbose

# Auto-detect domains directory
saber-domain list

# Explicit domains directory
saber-domain --domains-root /path/to/domains list
```

### `start` - Start Domain Services

**Automated Environment Generation** - The CLI automatically:
- Validates domain configuration and structure
- Generates all Docker Compose environment variables
- Checks port availability and suggests alternatives
- Validates or builds Docker images as needed
- Starts services with proper profiles and networking

```bash
# Basic start (server + client with auto-generated environment)
saber-domain start DOMAIN

# Rebuild all images first, then start
saber-domain start DOMAIN --rebuild

# Custom ports (with automatic conflict detection)
saber-domain start DOMAIN --rest-port 9000 --mcp-port 9001

# Custom profiles (validated against domain manifest)
saber-domain start DOMAIN --profiles server,client,docker-socket

# Custom logging level
saber-domain start DOMAIN --log-level DEBUG

# Dry run to see what would be done
saber-domain start DOMAIN --dry-run --rebuild

# Full example with all options
saber-domain start cybench \
  --rebuild \
  --profiles server,client,docker-socket \
  --rest-port 9000 \
  --mcp-port 9001 \
  --log-level DEBUG
```

**No Manual Configuration Required!** The CLI automatically generates:
```bash
# Traditional approach - manual .env editing:
DOMAIN=cybench
DOMAINS_ROOT=/path/to/domains
SERVER_IMAGE=saber/cybench/server:latest
CLIENT_IMAGE=saber/cybench/client:latest
REST_PORT=9000
MCP_PORT=9001
COMPOSE_PROFILES=server,client,docker-socket
LOG_LEVEL=DEBUG

# New approach - just run:
saber-domain start cybench --rebuild --profiles server,client,docker-socket --rest-port 9000 --mcp-port 9001 --log-level DEBUG
```

### `stop` - Stop Domain Services

```bash
# Stop domain services
saber-domain stop DOMAIN

# Dry run
saber-domain stop DOMAIN --dry-run
```

### `build` - Build Domain Images

```bash
# Build all images for a domain
saber-domain build DOMAIN

# Dry run to see build commands
saber-domain build DOMAIN --dry-run
```

### `validate` - Validate Domain Configuration

```bash
# Basic validation
saber-domain validate DOMAIN

# Detailed validation output
saber-domain validate DOMAIN --verbose
```

**Validation includes:**
- Domain manifest syntax and schema compliance
- Required directory structure (`server/`, `client/`, `docker/`)
- Docker image definitions and Dockerfile existence
- Profile definitions and requirements
- Port configuration

### `status` - Show Domain Status

```bash
# Status for all domains
saber-domain status

# Status for specific domain
saber-domain status DOMAIN

# Future: watch mode (not yet implemented)
saber-domain status DOMAIN --watch
```

## Global Options

```bash
# Auto-detect domains directory (searches common locations)
saber-domain COMMAND

# Explicit domains directory
saber-domain --domains-root /path/to/domains COMMAND

# Verbose output
saber-domain --verbose COMMAND

# Help
saber-domain --help
saber-domain COMMAND --help
```

## Domain Auto-Detection

The CLI automatically finds your domains directory by searching:
1. `--domains-root` option (highest priority)
2. `./domains` (current directory)
3. `../domains` (parent directory)  
4. `../../domains` (grandparent directory)

## Environment Generation Details

When you run `saber-domain start`, the CLI automatically:

### 1. **Domain Validation**
- Loads and validates `domain.yaml` against JSON schema
- Checks required directory structure
- Validates profile definitions
- Ensures domain slug matches directory name

### 2. **Environment Setup**
- Generates all required Docker Compose variables
- Sets domain-specific image tags from manifest
- Configures ports with conflict detection
- Sets up volume mounts and network configuration

### 3. **Image Management**
- Validates Docker images exist (or builds them with `--build`)
- Applies proper tagging with git metadata
- Includes domain-specific build args and labels

### 4. **Service Orchestration**
- Creates temporary environment file for Docker Compose
- Starts services with proper dependency ordering
- Monitors health checks and service readiness
- Cleans up temporary files automatically

## Profiles and Security

Domains declare their required profiles in `domain.yaml`:

```yaml
profiles:
  - name: docker-socket
    description: "Mount Docker socket for container management"
    justification: "Required for dynamic sandbox creation"
    required: true
```

The CLI validates requested profiles against the manifest:
- **Basic profiles**: `server`, `client` (always allowed)
- **Custom profiles**: Declared in domain manifest
- **Security profiles**: `docker-socket`, `privileged` (require explicit justification)

## Examples

### Development Workflow

```bash
# 1. List available domains
saber-domain list --verbose

# 2. Validate domain before starting
saber-domain validate cybench --verbose

# 3. Build and start with development settings
saber-domain start cybench \
  --build \
  --rest-port 8080 \
  --mcp-port 8081 \
  --log-level DEBUG \
  --dry-run  # Check first

# 4. Actually start (remove --dry-run)
saber-domain start cybench --build --rest-port 8080 --mcp-port 8081 --log-level DEBUG

# 5. Check status
saber-domain status cybench

# 6. Stop when done
saber-domain stop cybench
```

### Production Deployment

```bash
# Start with standard ports and required profiles
saber-domain start cybench --profiles server,client,docker-socket

# Check all domains
saber-domain status

# Validate configuration
saber-domain validate cybench
```

### CI/CD Pipeline

```bash
# Validate all domains
for domain in $(saber-domain list | grep -E '^\s+\w+' | awk '{print $1}'); do
  saber-domain validate "$domain"
done

# Build specific domain
saber-domain build cybench

# Test start with dry-run
saber-domain start cybench --dry-run --profiles server,client
```

## Error Handling

The CLI provides clear, actionable error messages:

```bash
$ saber-domain start cybench
Error: Domain 'cybench' configuration is invalid:
  - Port 8000 (REST) is already in use
  - Docker image 'saber/cybench/server:latest' for server not found. Use --build to create it.
  - Required profiles missing: ['docker-socket']

# Fix with:
$ saber-domain start cybench --build --rest-port 9000 --profiles server,client,docker-socket
```

## Migration from Legacy Scripts

If you were using the old `start_domain.py` script:

```bash
# Old approach:
./start_domain.py --domain cybench --action start --profiles server,client,docker-socket --build

# New approach:
saber-domain start cybench --profiles server,client,docker-socket --build
```

**Benefits of the new CLI:**
- No need to manage `.env` files manually
- Auto-detection of domains directory
- Integrated validation and error reporting
- Consistent with other SABER CLI tools
- Available as both `python -m saber.domain` and `saber-domain`
- Packaged resources work in all installation modes

## Troubleshooting

### Domain Not Found
```bash
saber-domain --domains-root /correct/path/to/domains list
```

### Port Conflicts
```bash
# Find available ports
saber-domain start DOMAIN --rest-port 9000 --mcp-port 9001
```

### Image Build Failures
```bash
# See build commands that would run
saber-domain build DOMAIN --dry-run

# Check domain structure
saber-domain validate DOMAIN --verbose
```

### Profile Issues  
```bash
# See available profiles
saber-domain validate DOMAIN --verbose

# Use correct profiles
saber-domain start DOMAIN --profiles server,client,required-profile
```

## Development

The domain CLI is built with:
- **Click** for CLI framework
- **Docker Compose** for orchestration  
- **JSON Schema** for manifest validation
- **importlib.resources** for packaged assets
- **Modular services** for testability

Key modules:
- `saber.domain.cli` - Click-based CLI interface
- `saber.domain.orchestrator` - Main orchestration logic
- `saber.domain.resources` - Packaged resource resolution
- `saber.domain.exceptions` - Domain-specific errors
