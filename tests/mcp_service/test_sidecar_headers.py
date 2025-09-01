import os
import contextlib
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from saber.client.mcp_service.main import create_app
from saber.client.mcp_service import main as sidecar_main
from saber.client.mcp_service.mcp_proxy import MCPResponse


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
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ready") is True


def test_session_register_and_list(client: TestClient):
    body = {"agent_id": "agent-1", "saber_session_id": "sess-1", "task_id": "task-1"}
    r = client.post("/admin/sessions", json=body)
    assert r.status_code == 200
    r = client.get("/admin/sessions")
    assert r.status_code == 200
    sessions = r.json().get("sessions", {})
    assert "agent-1" in sessions
    assert sessions["agent-1"]["saber_session_id"] == "sess-1"


def test_mcp_list_tools_requires_session_header(client: TestClient):
    # Register session
    client.post("/admin/sessions", json={"agent_id": "agent-2", "saber_session_id": "sess-2"})
    # Missing header should fail
    r = client.post("/mcp/list_tools", json={})
    assert r.status_code == 400
    # Wrong header should fail
    r = client.post("/mcp/list_tools", headers={"X-Agent-ID": "agent-2"}, json={})
    assert r.status_code == 400
    # Correct header succeeds
    r = client.post("/mcp/list_tools", headers={"X-Saber-Session-Id": "sess-2"}, json={})
    assert r.status_code == 200
    tools = r.json().get("result", [])
    assert isinstance(tools, list) and tools and tools[0]["name"] == "noop"


def test_mcp_list_tools_with_saber_session_header(client: TestClient):
    # Register session
    client.post("/admin/sessions", json={"agent_id": "agent-3", "saber_session_id": "sess-3"})
    # Call with preferred header (maps to agent id internally)
    r = client.post("/mcp/list_tools", headers={"X-Saber-Session-Id": "sess-3"}, json={})
    assert r.status_code == 200
    tools = r.json().get("result", [])
    assert isinstance(tools, list) and tools and tools[0]["name"] == "noop"


def test_mcp_call_tool_and_ping(client: TestClient):
    client.post("/admin/sessions", json={"agent_id": "agent-4", "saber_session_id": "sess-4"})
    r = client.post("/mcp/call_tool", headers={"X-Saber-Session-Id": "sess-4"},
                    json={"jsonrpc": "2.0", "method": "call_tool", "params": {"name": "noop", "arguments": {}}})
    assert r.status_code == 200
    assert r.json().get("result", {}).get("ok") is True
    r = client.post("/mcp/ping", headers={"X-Saber-Session-Id": "sess-4"})
    assert r.status_code == 200
    assert r.json().get("result", {}).get("pong") is True
