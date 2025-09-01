"""
Main MCP Service Application

FastAPI application that provides standard MCP protocol endpoints for agent containers
while routing requests to the appropriate SABER server sessions.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent_registry import AgentSessionRegistry
from .mcp_proxy import MCPProxy

logger = logging.getLogger(__name__)


# Request/Response Models
class SessionRegistrationRequest(BaseModel):
    """Request to register an agent container session."""

    agent_id: str = Field(..., description="Unique identifier for the agent container")
    saber_session_id: str = Field(..., description="SABER server session ID")
    task_id: Optional[str] = Field(None, description="Optional task ID for context")


class SessionRegistrationResponse(BaseModel):
    """Response for session registration."""

    success: bool
    message: str
    agent_id: str
    saber_session_id: str


class MCPListToolsResponse(BaseModel):
    """Response for MCP list_tools request."""

    jsonrpc: str = "2.0"
    result: Any
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class MCPCallToolRequest(BaseModel):
    """Request for MCP call_tool."""

    jsonrpc: str = "2.0"
    method: str = "call_tool"
    params: Dict[str, Any]
    id: Optional[str] = None


class MCPCallToolResponse(BaseModel):
    """Response for MCP call_tool request."""

    jsonrpc: str = "2.0"
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


# Global service instances
session_registry: Optional[AgentSessionRegistry] = None
mcp_proxy: Optional[MCPProxy] = None
cleanup_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for startup/shutdown."""
    global session_registry, mcp_proxy, cleanup_task

    # Startup
    logger.info("Starting MCP Service...")

    # Initialize core services
    session_registry = AgentSessionRegistry()
    # Pass the correct SABER MCP URL from app state
    saber_mcp_url = getattr(app.state, "saber_mcp_url", "http://localhost:8001")
    mcp_proxy = MCPProxy(session_registry, saber_mcp_url=saber_mcp_url)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    logger.info("✅ MCP Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MCP Service...")

    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    # Close shared HTTP client session
    try:
        if mcp_proxy:
            await mcp_proxy.aclose()
    except Exception as e:
        logger.warning(f"Error closing MCP proxy session: {e}")

    logger.info("✅ MCP Service shutdown complete")


async def periodic_cleanup() -> None:
    """Background task for periodic session cleanup."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            if session_registry:
                cleaned = await session_registry.cleanup_inactive_sessions()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} inactive sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


def create_app(saber_mcp_url: Optional[str] = None, timeout: float = 30.0) -> FastAPI:
    """
    Create and configure the MCP Service FastAPI application.

    Args:
        saber_mcp_url: URL of SABER MCP server
        timeout: Request timeout in seconds

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="SABER MCP Service",
        description="Shared MCP service for containerized agents",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store configuration for use in proxy
    import os

    app.state.saber_mcp_url = saber_mcp_url or os.environ.get("SABER_MCP_URL", "http://localhost:8001")
    if app.state.saber_mcp_url:
        app.state.saber_mcp_url = app.state.saber_mcp_url.rstrip("/")
    app.state.timeout = timeout

    async def update_proxy_config(request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Middleware to ensure MCP proxy has latest configuration."""
        global mcp_proxy  # noqa: F824
        if mcp_proxy and app.state.saber_mcp_url:
            mcp_proxy.saber_mcp_url = app.state.saber_mcp_url.rstrip("/")
        response = await call_next(request)
        return response

    app.middleware("http")(update_proxy_config)

    # Health endpoints
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        if not mcp_proxy:
            raise HTTPException(status_code=503, detail="MCP proxy not initialized")

        health_info = await mcp_proxy.health_check()

        if not health_info.get("saber_server_healthy", False):
            raise HTTPException(
                status_code=503, detail="SABER server not healthy", headers={"X-Health-Info": str(health_info)}
            )

        # Add status field for compatibility with tests
        return {"status": "healthy", **health_info}

    @app.get("/ready")
    async def readiness_check() -> Dict[str, Any]:
        """Readiness check endpoint."""
        if not session_registry or not mcp_proxy:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Readiness should not depend on server health
        return {"status": "ready", "message": "MCP Service is ready", "saber_mcp_url": app.state.saber_mcp_url}

    # Session management endpoints
    @app.post("/admin/sessions", response_model=SessionRegistrationResponse)
    async def register_session(request: SessionRegistrationRequest) -> SessionRegistrationResponse:
        """Register a new agent container session."""
        if not session_registry:
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        try:
            session = await session_registry.register_session(
                agent_id=request.agent_id, saber_session_id=request.saber_session_id, task_id=request.task_id
            )

            return SessionRegistrationResponse(
                success=True,
                message="Session registered successfully",
                agent_id=session.agent_id,
                saber_session_id=session.saber_session_id,
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to register session: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.delete("/admin/sessions/{agent_id}")
    async def unregister_session(agent_id: str) -> Dict[str, Any]:
        """Unregister an agent container session."""
        if not session_registry:
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        success = await session_registry.unregister_session(agent_id)

        if success:
            return {"success": True, "message": f"Session for agent {agent_id} unregistered"}
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    @app.get("/admin/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """List all active sessions."""
        if not session_registry:
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        sessions = await session_registry.list_active_sessions()

        return {
            "sessions": {
                agent_id: {
                    "saber_session_id": session.saber_session_id,
                    "task_id": session.task_id,
                    "registered_at": session.registered_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "is_active": session.is_active,
                }
                for agent_id, session in sessions.items()
            }
        }

    # Standard MCP Protocol Endpoints
    async def get_agent_id_from_required_session_header(request: Request) -> str:
        """Resolve agent_id strictly from X-Saber-Session-Id.

        - If header is missing: 400 Bad Request
        - If session not registered: 404 Not Found
        """
        if not session_registry:
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        saber_session_id = request.headers.get("X-Saber-Session-Id")
        if not saber_session_id:
            raise HTTPException(status_code=400, detail="Missing required header: X-Saber-Session-Id")

        agent_id = await session_registry.get_agent_id_by_saber_session(saber_session_id)
        if not agent_id:
            raise HTTPException(status_code=404, detail="Session not registered: X-Saber-Session-Id")

        return agent_id

    @app.post("/mcp/list_tools", response_model=MCPListToolsResponse)
    async def list_tools(request: Request) -> MCPListToolsResponse:
        """Standard MCP list_tools endpoint."""
        if not mcp_proxy:
            raise HTTPException(status_code=503, detail="MCP proxy not initialized")
        agent_id = await get_agent_id_from_required_session_header(request)
        response = await mcp_proxy.list_tools(agent_id)
        if response.error:
            return MCPListToolsResponse(result=None, error=response.error)
        return MCPListToolsResponse(result=response.result)

    @app.post("/mcp/call_tool", response_model=MCPCallToolResponse)
    async def call_tool(request: Request, mcp_request: MCPCallToolRequest) -> MCPCallToolResponse:
        """Standard MCP call_tool endpoint."""
        if not mcp_proxy:
            raise HTTPException(status_code=503, detail="MCP proxy not initialized")
        agent_id = await get_agent_id_from_required_session_header(request)
        # Extract tool name and arguments from MCP request
        params = mcp_request.params
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name:
            return MCPCallToolResponse(
                error={"code": -32602, "message": "Missing required parameter 'name'"},
                id=mcp_request.id,
            )
        response = await mcp_proxy.call_tool(agent_id, tool_name, arguments)
        return MCPCallToolResponse(result=response.result, error=response.error, id=mcp_request.id)

    @app.post("/mcp/list_resources")
    async def list_resources(request: Request) -> JSONResponse:
        """Standard MCP list_resources endpoint."""
        if not mcp_proxy:
            raise HTTPException(status_code=503, detail="MCP proxy not initialized")
        agent_id = await get_agent_id_from_required_session_header(request)
        response = await mcp_proxy.list_resources(agent_id)
        if response.error:
            return JSONResponse(content={"jsonrpc": "2.0", "result": None, "error": response.error})
        return JSONResponse(content={"jsonrpc": "2.0", "result": response.result})

    @app.post("/mcp/ping")
    async def ping(request: Request) -> JSONResponse:
        """Standard MCP ping endpoint."""
        if not mcp_proxy:
            raise HTTPException(status_code=503, detail="MCP proxy not initialized")
        agent_id = await get_agent_id_from_required_session_header(request)
        response = await mcp_proxy.ping(agent_id)
        return JSONResponse(content={"jsonrpc": "2.0", "result": response.result, "error": response.error})

    return app


# CLI entry point for development/testing
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")


# Create default app instance for imports
app = create_app()


def start_sidecar_service(host: str = "0.0.0.0", port: int = 8002, saber_mcp_url: Optional[str] = None) -> None:
    """Start the MCP sidecar service."""
    import uvicorn

    service_app = create_app(saber_mcp_url=saber_mcp_url)
    uvicorn.run(service_app, host=host, port=port, log_level="info")
