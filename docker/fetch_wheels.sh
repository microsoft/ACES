#!/usr/bin/env bash
# Refresh the vendored PyPI wheels for the saber/sandbox base image.
#
# The base sandbox (Dockerfile.saber_sandbox) installs these Python packages from
# PyPI so builds always pick up the latest versions, and uses docker/wheels/ only
# as a FALLBACK when container egress to files.pythonhosted.org (PyPI's CDN) is
# blocked (the host can still reach PyPI). The wheels are NOT committed — run
# this on such a host before building to populate them.
#
# Wheels must target the base image interpreter: CPython 3.11 / linux x86_64.
set -euo pipefail

WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wheels"
mkdir -p "$WHEELS_DIR"
rm -f "$WHEELS_DIR"/*.whl

python3 -m pip download \
    --only-binary=:all: \
    --python-version 311 \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_17_x86_64 \
    -d "$WHEELS_DIR" \
    requests \
    "github-copilot-sdk>=0.3.0" \
    claude-code-sdk

echo "Refreshed wheels in $WHEELS_DIR:"
ls -1 "$WHEELS_DIR"
