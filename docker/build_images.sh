#!/bin/bash
# SABER Docker Images Build Script
#
# Builds all necessary Docker images for the SABER container architecture.
# Supports both development and production configurations.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY="${SABER_REGISTRY:-local}"
VERSION="${SABER_VERSION:-latest}"
BUILD_TYPE="${BUILD_TYPE:-dev}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build an image
build_image() {
    local dockerfile="$1"
    local image_name="$2"
    local context="$3"
    local build_args="${4:-}"

    log_info "Building image: $image_name"
    log_info "Dockerfile: $dockerfile"
    log_info "Context: $context"

    local full_image_name
    if [[ "$REGISTRY" == "local" ]]; then
        full_image_name="$image_name:$VERSION"
    else
        full_image_name="$REGISTRY/$image_name:$VERSION"
    fi

    local docker_cmd="docker build"
    docker_cmd="$docker_cmd -f $dockerfile"
    docker_cmd="$docker_cmd -t $full_image_name"

    # Add build args if provided
    if [[ -n "$build_args" ]]; then
        docker_cmd="$docker_cmd $build_args"
    fi

    # Add build type specific args
    if [[ "$BUILD_TYPE" == "prod" ]]; then
        docker_cmd="$docker_cmd --no-cache"
    fi

    docker_cmd="$docker_cmd $context"

    log_info "Executing: $docker_cmd"

    if eval "$docker_cmd"; then
        log_success "Built image: $full_image_name"
        return 0
    else
        log_error "Failed to build image: $full_image_name"
        return 1
    fi
}

# Function to tag image for different registries
tag_image() {
    local source_image="$1"
    local target_image="$2"

    log_info "Tagging image: $source_image -> $target_image"

    if docker tag "$source_image" "$target_image"; then
        log_success "Tagged image: $target_image"
    else
        log_error "Failed to tag image: $target_image"
        return 1
    fi
}

# Function to push image to registry
push_image() {
    local image_name="$1"

    if [[ "$REGISTRY" == "local" ]]; then
        log_info "Skipping push for local registry"
        return 0
    fi

    log_info "Pushing image: $image_name"

    if docker push "$image_name"; then
        log_success "Pushed image: $image_name"
    else
        log_error "Failed to push image: $image_name"
        return 1
    fi
}

# Main build function
main() {
    log_info "SABER Docker Images Build"
    log_info "=========================="
    log_info "Registry: $REGISTRY"
    log_info "Version: $VERSION"
    log_info "Build Type: $BUILD_TYPE"
    log_info "Project Root: $PROJECT_ROOT"
    echo

    # Ensure we're in the project root
    cd "$PROJECT_ROOT"

    # Build images
    local build_failed=0

    # 1. Build MCP Sidecar Image
    log_info "🔧 Building MCP Sidecar Image..."
    if build_image \
        "docker/Dockerfile.mcp-service" \
        "saber-mcp-sidecar" \
        "." \
        "--build-arg BUILD_TYPE=$BUILD_TYPE"; then
        log_success "MCP Sidecar image built successfully"
    else
        log_error "MCP Sidecar image build failed"
        build_failed=1
    fi
    echo

    # 2. Build Agent Runner Base Image
    log_info "🤖 Building Agent Runner Base Image..."
    if build_image \
        "docker/Dockerfile.agent-runner" \
        "saber-agent-runner" \
        "." \
        "--build-arg BUILD_TYPE=$BUILD_TYPE"; then
        log_success "Agent Runner base image built successfully"
    else
        log_error "Agent Runner base image build failed"
        build_failed=1
    fi
    echo

    # 3. Build additional sandbox images if they exist
    if [[ -f "docker/Dockerfile.python_sandbox" ]]; then
        log_info "🐍 Building Python Sandbox Image..."
        if build_image \
            "docker/Dockerfile.python_sandbox" \
            "saber-python-sandbox" \
            "docker"; then
            log_success "Python Sandbox image built successfully"
        else
            log_error "Python Sandbox image build failed"
            build_failed=1
        fi
        echo
    fi

    # Tag images for different environments if needed
    if [[ "$BUILD_TYPE" == "prod" && "$REGISTRY" != "local" ]]; then
        log_info "🏷️  Tagging images for production..."

        # Tag with latest
        tag_image "saber-mcp-sidecar:$VERSION" "$REGISTRY/saber-mcp-sidecar:latest"
        tag_image "saber-agent-runner:$VERSION" "$REGISTRY/saber-agent-runner:latest"

        # Push images
        log_info "📤 Pushing images to registry..."
        push_image "$REGISTRY/saber-mcp-sidecar:$VERSION"
        push_image "$REGISTRY/saber-mcp-sidecar:latest"
        push_image "$REGISTRY/saber-agent-runner:$VERSION"
        push_image "$REGISTRY/saber-agent-runner:latest"
    fi

    # Summary
    echo
    log_info "Build Summary"
    log_info "============="

    if [[ $build_failed -eq 0 ]]; then
        log_success "All images built successfully!"

        log_info "Built images:"
        docker images | grep -E "(saber-mcp-sidecar|saber-agent-runner)" | head -10

        echo
        log_info "🚀 Ready to use SABER container architecture!"
        log_info "   Start sidecar: docker run -p 8002:8002 saber-mcp-sidecar:$VERSION"
        log_info "   Agent runner available: saber-agent-runner:$VERSION"
    else
        log_error "Some images failed to build. Check the logs above."
        exit 1
    fi
}

# Help function
show_help() {
    echo "SABER Docker Images Build Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -r, --registry   Set registry (default: local)"
    echo "  -v, --version    Set version tag (default: latest)"
    echo "  -t, --type       Set build type: dev|prod (default: dev)"
    echo "  --push           Push images to registry after build"
    echo
    echo "Environment Variables:"
    echo "  SABER_REGISTRY   Registry URL (default: local)"
    echo "  SABER_VERSION    Image version tag (default: latest)"
    echo "  BUILD_TYPE       Build type: dev|prod (default: dev)"
    echo
    echo "Examples:"
    echo "  $0                                    # Build all images locally"
    echo "  $0 -r my-registry.com -v v1.0.0      # Build and tag for registry"
    echo "  $0 -t prod --push                    # Production build and push"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            if [[ "$BUILD_TYPE" != "dev" && "$BUILD_TYPE" != "prod" ]]; then
                log_error "Invalid build type: $BUILD_TYPE. Must be 'dev' or 'prod'"
                exit 1
            fi
            shift 2
            ;;
        --push)
            export PUSH_IMAGES=1
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check dependencies
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Run main function
main
