#!/usr/bin/env bash
# Vendor the agent CLI npm tarballs for the saber/sandbox base image.
#
# You only need this if BOTH are true:
#   1. you want the `copilot` / `claude_code` agent harnesses, and
#   2. container builds on this machine cannot reach registry.npmjs.org.
#
# The default `react` harness needs none of this - the image builds fine without
# it. Run this on a host that CAN reach the npm registry, then rebuild the image;
# Dockerfile.saber_sandbox falls back to these tarballs automatically.
#
# The tarballs are NOT committed (the platform package is ~280MB and would go
# stale). They are downloaded straight over HTTPS, so no local npm is required.
set -euo pipefail

NPM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/npm"
# Both CLIs ship their actual binary in a platform-specific optionalDependency.
# These suffixes match the base image platform (python:3.11-slim => linux glibc x64).
COPILOT_PLATFORM="${SABER_COPILOT_PLATFORM:-@github/copilot-linux-x64}"
CLAUDE_PLATFORM="${SABER_CLAUDE_PLATFORM:-@anthropic-ai/claude-code-linux-x64}"

mkdir -p "$NPM_DIR"
rm -f "$NPM_DIR"/*.tgz

python3 - "$NPM_DIR" "$COPILOT_PLATFORM" "$CLAUDE_PLATFORM" <<'PY'
import json, sys, urllib.parse, urllib.request
from pathlib import Path

out_dir = Path(sys.argv[1])
platform_pkgs = {"@github/copilot": sys.argv[2], "@anthropic-ai/claude-code": sys.argv[3]}


def metadata(spec: str) -> dict:
    url = "https://registry.npmjs.org/" + urllib.parse.quote(spec, safe="")
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def download(spec: str, version: str) -> None:
    tarball = metadata(spec)["versions"][version]["dist"]["tarball"]
    dest = out_dir / tarball.rsplit("/", 1)[-1]
    print(f"  ↓ {spec}@{version}")
    urllib.request.urlretrieve(tarball, dest)
    print(f"    ✓ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


for cli, platform_pkg in platform_pkgs.items():
    meta = metadata(cli)
    version = meta["dist-tags"]["latest"]
    download(cli, version)
    # Without the platform package the CLI reports "native binary not installed" /
    # "no platform package found"; it is pinned to the same version as the CLI.
    optional = meta["versions"][version].get("optionalDependencies", {})
    download(platform_pkg, optional.get(platform_pkg, version))

download("detect-libc", metadata("detect-libc")["dist-tags"]["latest"])
PY

echo
echo "Vendored npm tarballs in $NPM_DIR:"
ls -1sh "$NPM_DIR"
