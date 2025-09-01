# Integration Tests for Container-Based MCP Sidecar Architecture

This directory contains end-to-end integration tests for the new container-based MCP sidecar architecture implemented in Phase 4 of the MCP sidecar implementation plan.

## Test Files

### `test_container_e2e_integration.py`
Full end-to-end integration test that validates:
- Mock SABER server startup (REST API + MCP server)
- Sidecar container lifecycle management
- Agent container execution with MCP tool calls
- Complete cleanup and termination

### `test_harness_container_integration.py`
Harness-level integration tests focusing on:
- Container mode initialization vs embedded mode
- Episode execution through container architecture
- Error handling and cleanup
- Resource management

### `mock_test_agent.py`
Lightweight mock agent for testing that demonstrates:
- Standard agent interface (`run` method)
- MCP tool discovery and execution via sidecar
- Proper result formatting for container collection

## Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and excluded from default test runs.

### Run all integration tests:
```bash
uv run pytest tests/integration/ -m integration
```

### Run specific integration test files:
```bash
# E2E container tests
uv run pytest tests/integration/test_container_e2e_integration.py -m integration

# Harness integration tests
uv run pytest tests/integration/test_harness_container_integration.py -m integration
```

### Run with verbose output:
```bash
uv run pytest tests/integration/ -m integration -v
```

### Run only unit tests (default):
```bash
uv run pytest  # Excludes integration tests by default
```

## Test Dependencies

Integration tests require:
- Docker daemon running locally
- Container images built (`saber-mcp-sidecar`, `saber-agent-runner`)
- Network connectivity for container communication

## Test Architecture

The integration tests use a layered approach:

1. **Mock SABER Server**: Lightweight FastAPI server providing REST and MCP endpoints
2. **Real Containers**: Actual sidecar and agent containers from the implementation
3. **Controlled Environment**: Isolated Docker networks and proper cleanup
4. **Multi-Channel Validation**: Tests termination via REST API, container status, and MCP connections

## Building Required Images

Before running integration tests, build the required container images:

```bash
cd docker/
./build_images.sh
```

This creates:
- `saber-mcp-sidecar:latest` (295MB)
- `saber-agent-runner:latest` (458MB)

## Test Configuration

Tests are configured via `pyproject.toml`:
- Integration marker: `integration: marks tests as integration tests`
- Default exclusion: `-m 'not integration'`
- Async support: `pytest-asyncio`
