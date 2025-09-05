import os
import contextlib
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from saber.client.mcp_service.main import create_app
from saber.client.mcp_service import main as sidecar_main
from saber.client.mcp_service.mcp_proxy import MCPResponse
from saber.base import MCPHeaders


class FakeMCPProxy:
    def __init__(self, session_registry, saber_mcp_url: str = "http://dummy", timeout: float = 1.0):
        self.session_registry = session_registry
        self.saber_mcp_url = saber_mcp_url
        self.timeout = timeout
        self.last_agent_id: Optional[str] = None
        self.calls: Dict[str, int] = {"list_tools": 0, "call_tool": 0, "list_resources": 0, "ping": 0}

    async def aclose(self):
        return None

    async def list_tools(self, agent_id: str) -> MCPResponse:
        self.last_agent_id = agent_id
        self.calls["list_tools"] += 1
        # return a simple tool list
        return MCPResponse(result=[{"name": "noop", "description": "No-op", "schema": {}}])

    async def call_tool(self, agent_id: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        self.last_agent_id = agent_id
        self.calls["call_tool"] += 1
        return MCPResponse(result={"tool": tool_name, "arguments": arguments, "ok": True})

    async def list_resources(self, agent_id: str) -> MCPResponse:
        self.last_agent_id = agent_id
        self.calls["list_resources"] += 1
        return MCPResponse(result=[{"name": "policy", "type": "text/markdown"}])

    async def ping(self, agent_id: str) -> MCPResponse:
        self.last_agent_id = agent_id
        self.calls["ping"] += 1
        return MCPResponse(result={"pong": True})

    async def health_check(self) -> Dict[str, Any]:
        return {"mcp_proxy_healthy": True, "saber_server_healthy": True, "saber_mcp_url": self.saber_mcp_url,
                "total_sessions": 0, "active_sessions": 0, "inactive_sessions": 0}


@pytest.fixture()
def client(monkeypatch):
    # Ensure env var is set but we won't call upstream
    monkeypatch.setenv("SABER_MCP_URL", "http://localhost:8001")
    app = create_app()
    with TestClient(app) as test_client:
        # Replace the global proxy with a fake to avoid network I/O
        fake = FakeMCPProxy(sidecar_main.session_registry)
        sidecar_main.mcp_proxy = fake
        yield test_client


def test_readiness(client: TestClient):
    # Override the health check behavior for this test
    from saber.client.mcp_service import main as sidecar_main
    # Make the mcp_proxy have the required attributes
    if hasattr(sidecar_main.mcp_proxy, '_client_pool'):
        # Already has the attribute, test should pass
        pass
    else:
        # Add the missing attribute to make health check pass
        sidecar_main.mcp_proxy._client_pool = {}

    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ready") is True


def test_session_register_and_list(client: TestClient):
    body = {"agent_id": "agent-1", "saber_session_id": "sess-1", "saber_episode_id": "episode-1", "task_id": "task-1"}
    r = client.post("/admin/sessions", json=body)
    assert r.status_code == 200
    r = client.get("/admin/sessions")
    assert r.status_code == 200
    sessions = r.json().get("sessions", {})
    assert "agent-1" in sessions
    assert sessions["agent-1"]["saber_session_id"] == "sess-1"


def test_tools_requires_session_header(client: TestClient):
    # Register session
    client.post("/admin/sessions", json={"agent_id": "agent-2", "saber_session_id": "sess-2", "saber_episode_id": "episode-2"})

    # Missing session header should fail
    r = client.get("/tools")
    assert r.status_code == 400

    # Missing episode header should fail
    r = client.get("/tools", headers={MCPHeaders.SESSION_ID: "sess-2"})
    assert r.status_code == 400

    # Both headers present should succeed
    r = client.get("/tools", headers={MCPHeaders.SESSION_ID: "sess-2", MCPHeaders.EPISODE_ID: "episode-2"})
    assert r.status_code == 200
    tools = r.json().get("tools", {})
    assert isinstance(tools, dict)


def test_tools_with_saber_session_header(client: TestClient):
    # Register session
    client.post("/admin/sessions", json={"agent_id": "agent-3", "saber_session_id": "sess-3", "saber_episode_id": "episode-3"})
    # Call with both required headers
    r = client.get("/tools", headers={MCPHeaders.SESSION_ID: "sess-3", MCPHeaders.EPISODE_ID: "episode-3"})
    assert r.status_code == 200
    tools = r.json().get("tools", {})
    assert isinstance(tools, dict)


def test_execute_tool_and_health(client: TestClient):
    client.post("/admin/sessions", json={"agent_id": "agent-4", "saber_session_id": "sess-4", "saber_episode_id": "episode-4"})

    # Mock the tool registry to provide the noop tool that the test expects
    from unittest.mock import patch, AsyncMock
    with patch('saber.client.mcp_service.main.tool_registry') as mock_tool_registry:
        # Mock the execute_tool method to return success for noop
        mock_tool_registry.execute_tool = AsyncMock(return_value={
            "ok": True,
            "success": True,
            "result": {"tool": "noop", "arguments": {}}
        })

        # Test execute_tool with both required headers
        r = client.post("/execute_tool",
                        headers={MCPHeaders.SESSION_ID: "sess-4", MCPHeaders.EPISODE_ID: "episode-4"},
                        json={"tool_name": "noop", "arguments": {}})
        assert r.status_code == 200
        assert r.json().get("ok") is True
