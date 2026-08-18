# SABER → GitHub Migration Checklist

**Codename note:** external name **ACES**, internal codename **SABER**; the package/CLI is `saber`. This repository is the **library**.

**Goal:** make GitHub the permanent home for the SABER library and retire the Azure DevOps mirror.

| Repository | GitHub (permanent home) | Azure DevOps (deprecated) |
|---|---|---|
| **Library** (this repo) | [microsoft/ACES](https://github.com/microsoft/ACES) | [SABER](https://dev.azure.com/MSECAIModels/Benchmarking/_git/SABER) |
| **Benchmarks** | [microsoft/ACESEvals](https://github.com/microsoft/ACESEvals) | [oss_saber](https://dev.azure.com/MSECAIModels/Benchmarking/_git/oss_saber) |

**Azure DevOps mirrors are read-only after 2026-09-30 and will then be retired.** Do all new work on GitHub.

---

## Done on this branch (`deprecate-ado-migrate-to-github`)

### Documentation → GitHub-only
- [x] `README.md`: ADO deprecation banner (wrapped in `<!-- ADO-ONLY:START/END -->` so the GitHub mirror can strip it) + "Repositories (GitHub is the permanent home)" section; removed the "ADO users — required setup step" note.
- [x] `.github/copilot-instructions.md`: repos preamble now states GitHub is the permanent home and ADO is deprecated (read-only after 2026-09-30).

### Configuration → GitHub-only
- [x] `pyproject.toml` `[tool.uv.sources]`: `inspect-ai` sourced only from the GitHub ACESEvals fork (branch `inspect-ai/dev/aces_integration`); removed the ADO and scaffold-comment alternatives and strengthened the **REQUIRED fork** note (it powers the `copilot`/`claude_code` harnesses and tool_call limits — do not switch to upstream inspect_ai).
- [x] Dependency floors raised for the current inspect-ai fork harnesses and `uv.lock` refreshed:
  - `openai>=2.40.0` (react/copilot harnesses need the Responses API) — locked `2.53.0`.
  - `anthropic>=0.105.0` (claude_code harness `generate_anthropic` bridge path) — locked `0.121.0`.

### Sandbox images — react-only base + agent-CLI variant
- [x] Split the single sandbox image in two, selected automatically from `-T agent=` (users choose nothing):
  - `saber/sandbox:latest` (~320MB) — Python, uv, requests: what the default `react` harness needs and nothing more, so it builds with **no npm access at all**.
  - `saber/sandbox:agents` (~2.9GB) — adds Node, the Copilot / Claude Code CLIs and the Python agent SDKs, layered on the base.
- [x] Domain images build FROM the required variant via the `SABER_BASE_IMAGE` build arg. Each variant stamps a `saber.base_variant` label that derived images inherit, so a domain image without the CLIs is rebuilt automatically when a CLI harness is requested. Images that do not derive from the saber base (the excytin databases, say) are left alone rather than rebuilt every run.
- [x] Agent CLI install is layered: npm registry → vendored `docker/npm/` tarballs → clear error. Both CLIs ship their real binary in a platform-specific `optionalDependency` (`@github/copilot-linux-x64`, `@anthropic-ai/claude-code-linux-x64`), so `docker/fetch_npm.sh` vendors those too. A build-time warm-up bakes the one-time platform unpack into the image (first CLI invocation 5.3s → 1.2s), which matters because every eval sample gets a fresh sandbox.

### Harness fixes
- [x] **Copilot SDK:** install `github-copilot-sdk>=0.3.0` (resolves to public **1.0.9**), replacing the non-existent `==0.3.0` pin. The stale base image shipped `0.1.32`, whose `CopilotClient.create_session()` lacks the `model` kwarg the bridge passes. `1.0.9`'s `create_session` accepts `model`, `reasoning_effort`, `session_id`, etc.
- [x] **Claude native binary:** run `install.cjs` explicitly after the `npm install --ignore-scripts` (the flag skips the postinstall that fetches the native binary). Fixes the runtime `claude: native binary not installed` error.
- [x] **Copilot CLI:** the `github-copilot-sdk` downloads its own copy of the CLI at session start, which fails inside the network-isolated eval sandbox. The image already installs the CLI via npm, so `COPILOT_CLI_PATH=/usr/bin/copilot` points the SDK at it — no download, no CDN egress needed at build, and only one copy of the CLI. Verified end-to-end: excytin copilot harness scores 1.000 (previously it failed at session creation).
- [x] **Release pipeline build context:** `docker-build-steps.yml` built the sandbox with `buildContext: .`, but the wheels fallback adds `COPY wheels/`, which only exists at `docker/wheels/` (`COPY failed: stat wheels/: file does not exist`). Only `release.yml` builds images, so this would have stayed silent until a release. The sandbox build now uses `docker/` as its context, matching how saber builds the image locally.
- [x] **Offline-capable Python install:** the SDKs install from PyPI so builds always pick up the latest versions. Where container egress to `files.pythonhosted.org` is blocked, run `docker/fetch_wheels.sh` on the host and the build falls back to `docker/wheels/`. Those wheels are **not committed** (they go stale and bloat the repo) — only an empty placeholder dir is tracked, and the package ships the placeholder + fetch script rather than the wheels.

### Verification
- [x] Base image builds offline (pip) and yields `github-copilot-sdk 1.0.9` (with `create_session(model=…)`) and `claude 2.1.207`.
- [x] Copilot harness smoke (excytin): the `create_session()` error is resolved — the runner creates the session and proceeds to task execution.
- [x] Claude Code harness smoke (excytin): the "native binary not installed" error is resolved — the CLI starts and reaches model-proxy calls.
- [x] **Full end-to-end run on fresh clones of both migration branches** (gpt-5.4, `--reasoning-effort high`): 3 domains × 3 harnesses, plus excytin normal *and* insane at 20 samples per incident — **58 cells, 0 failures, ~20h compute**, with the offline artifact path exercised throughout on a network that blocks npm and PyPI's CDN.
  - excytin normal: react 0.915 / claude_code 0.881 / copilot 0.862
  - excytin insane: react 0.818 / claude_code 0.773 / copilot 0.714
  - cti_realm (20 samples): react 0.515 / copilot 0.462 / claude_code 0.175
  - cybench: react 0.500 / claude_code 0.667 / copilot 0.000 — note the domain ships a **single task**, so each cell is one sample and is not comparably powered.

---

## Follow-ups

- [ ] **Merge order matters.** The benchmark repo resolves `saber` from GitHub `ACES@main`, so this library's changes must land there **before** the benchmark repo's lock is bumped to pick them up. Merging the benchmark side first is safe but inert: its domain Dockerfiles default `SABER_BASE_IMAGE` to `saber/sandbox:latest`, so an older saber simply keeps the previous single-image behaviour.

- [x] **Domain sandbox Dockerfiles that pip-install** now build on restricted networks (benchmark repo): excytin, cti_realm, and cybench all install from PyPI first and fall back to `docker/wheels/` (populated on demand via each domain's `fetch_wheels.sh`) only when the CDN is unreachable.
- [x] **Rebuild derived domain sandboxes** from the new base: `saber/excytin/sandbox` rebuilt + verified (copilot-sdk 1.0.9, pymysql, sqlalchemy, claude 2.1.207, `COPILOT_CLI_PATH` set); cti_realm rebuilds via the same fallback path.
- [x] **Copilot runner non-proxied fetch** (the `MS.txt` failure) traced to the SDK's CLI download and fixed by pointing `COPILOT_CLI_PATH` at the npm-installed CLI.
- [x] **cti_realm build-arg path** replaced by the automatic PyPI-first fallback (no `SABER_OFFLINE_WHEELS` flag needed).
- [ ] **Port the release pipelines to GitHub Actions — required before ADO is retired.** `.pipelines/` builds `saber-server` + `saber-sandbox` with the ADO `Docker@2` task and pushes them to the internal ACR `benchmarking-spear-stg-acr`, using ADO artifact feeds and `System.AccessToken`. None of that exists on GitHub, so image releases stop when ADO goes read-only. Needs a registry decision (GHCR vs. keeping ACR) before it can be written.
- [ ] **From-scratch base rebuild on restricted networks:** `registry.npmjs.org` and the Claude CDN are blocked to containers on some networks, and prefetching npm needs host node/npm. The official images are built by CI with unrestricted egress, so this only affects local rebuilds — build the base where egress exists.
- [ ] **(Deferred, intentional) Python `claude-code-sdk`** is kept in the sandbox (may be used later); not slimmed out.

---

## ADO retirement
- Announce the move; ADO `SABER`/`oss_saber` become read-only after **2026-09-30** and are then retired.
- Point clones, pipelines, and references at GitHub (`microsoft/ACES` + `microsoft/ACESEvals`).
