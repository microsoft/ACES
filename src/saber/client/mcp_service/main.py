"""
Main MCP Service Application

FastAPI application that provides standard MCP protocol endpoints for agent containers
while routing requests to the appropriate SABER server sessions.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent_registry import AgentSessionRegistry
from .mcp_proxy import MCPProxy
from .tool_registry import ToolRegistry


def setup_logging() -> None:
    """Configure clean Python logging for sidecar container."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure clean console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Use same format as agent containers
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Set up root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Disable uvicorn's default loggers to prevent double logging
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Initialize logging before creating logger
setup_logging()
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


class ToolExecutionRequest(BaseModel):
    """Request for tool execution via function injection."""

    tool_name: str = Field(..., description="Name of tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: Optional[int] = Field(None, description="Execution timeout override")


class ToolExecutionResponse(BaseModel):
    """Response for tool execution."""

    success: bool
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


# Global service instances
session_registry: Optional[AgentSessionRegistry] = None
mcp_proxy: Optional[MCPProxy] = None
tool_registry: Optional[ToolRegistry] = None
cleanup_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for startup/shutdown."""
    global session_registry, mcp_proxy, tool_registry, cleanup_task

    # Startup
    logger.info("🚀 Starting MCP Service...")

    try:
        # Initialize core services
        logger.info("📋 Initializing session registry...")
        session_registry = AgentSessionRegistry()
        logger.info("✅ Session registry initialized")

        # Get configuration from app state
        saber_mcp_url = getattr(app.state, "saber_mcp_url", "http://localhost:8001")
        harness_url = getattr(app.state, "harness_url", "http://localhost:8000")
        logger.info(f"🔧 Configuration: saber_mcp_url={saber_mcp_url}, harness_url={harness_url}")

        logger.info(" Initializing MCP proxy...")
        mcp_proxy = MCPProxy(session_registry, saber_mcp_url=saber_mcp_url)
        logger.info("✅ MCP proxy initialized")

        # Initialize tool registry
        logger.info("🛠️  Initializing tool registry...")
        tool_registry = ToolRegistry(mcp_proxy)
        logger.info("✅ Tool registry initialized")

        # Start background cleanup task
        logger.info("🧹 Starting background cleanup task...")
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info("✅ Background cleanup task started")

        logger.info("🎉 MCP Service started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start MCP Service: {e}", exc_info=True)
        raise

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

    # Cleanup tool registry
    if tool_registry:
        # Cancel all refresh tasks
        for cache_key in list(tool_registry._refresh_tasks.keys()):
            session_id, episode_id = cache_key
            await tool_registry.cleanup_session_episode(session_id, episode_id)

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


def create_app(
    saber_mcp_url: Optional[str] = None, timeout: float = 30.0, harness_url: Optional[str] = None
) -> FastAPI:
    """
    Create and configure the MCP Service FastAPI application.

    Args:
        saber_mcp_url: URL of SABER MCP server
        timeout: Request timeout in seconds
        harness_url: URL of SABER harness for progress reporting

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
    app.state.harness_url = (
        harness_url or os.environ.get("HARNESS_URL") or os.environ.get("SABER_HARNESS_URL", "http://localhost:8000")
    )

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
        """Readiness check endpoint with detailed diagnostics."""
        try:
            # Check if global objects exist
            if not session_registry:
                logger.error("❌ Health check failed: session_registry is None")
                raise HTTPException(status_code=503, detail="Session registry not initialized")

            if not mcp_proxy:
                logger.error("❌ Health check failed: mcp_proxy is None")
                raise HTTPException(status_code=503, detail="MCP proxy not initialized")

            # Check if MCP proxy has required attributes
            if not hasattr(mcp_proxy, "_client_pool"):
                logger.error("❌ Health check failed: mcp_proxy missing _client_pool attribute")
                raise HTTPException(status_code=503, detail="MCP proxy incomplete initialization")

            # Test basic proxy functionality
            try:
                proxy_status = await mcp_proxy.health_check()
                logger.debug(f"✅ MCP proxy health check passed: {proxy_status}")
            except Exception as e:
                logger.error(f"❌ MCP proxy health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"MCP proxy health check failed: {e}")

            logger.debug("✅ All health checks passed")
            return {
                "status": "ready",
                "ready": True,
                "message": "MCP Service is ready",
                "saber_mcp_url": app.state.saber_mcp_url,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            raise HTTPException(status_code=503, detail=f"Health check error: {e}")

    # Session management endpoints
    @app.post("/admin/sessions", response_model=SessionRegistrationResponse)
    async def register_session(request: SessionRegistrationRequest) -> SessionRegistrationResponse:
        """Register a new agent container session."""
        logger.info(
            f"🔐 Session registration request: agent_id={request.agent_id}, "
            f"saber_session_id={request.saber_session_id}, task_id={request.task_id}"
        )

        if not session_registry:
            logger.error("❌ Session registration failed: Session registry not initialized")
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        try:
            session = await session_registry.register_session(
                agent_id=request.agent_id, saber_session_id=request.saber_session_id, task_id=request.task_id
            )

            logger.info(f"✅ Session registered successfully: {session.agent_id} → {session.saber_session_id}")
            return SessionRegistrationResponse(
                success=True,
                message="Session registered successfully",
                agent_id=session.agent_id,
                saber_session_id=session.saber_session_id,
            )

        except ValueError as e:
            logger.error(f"❌ Session registration failed (validation): {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"❌ Session registration failed (internal): {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.delete("/admin/sessions/{agent_id}")
    async def unregister_session(agent_id: str) -> Dict[str, Any]:
        """Unregister an agent container session."""
        logger.info(f"🔓 Session unregistration request: agent_id={agent_id}")

        if not session_registry:
            logger.error("❌ Session unregistration failed: Session registry not initialized")
            raise HTTPException(status_code=503, detail="Session registry not initialized")

        success = await session_registry.unregister_session(agent_id)

        if success:
            logger.info(f"✅ Session unregistered successfully: {agent_id}")
            return {"success": True, "message": f"Session for agent {agent_id} unregistered"}
        else:
            logger.error(f"❌ Session unregistration failed: Agent {agent_id} not found")
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

    # Function Injection Endpoints for New Architecture
    @app.get("/tools")
    async def get_tools_metadata(request: Request) -> JSONResponse:
        """Get available tools for the requesting session+episode."""
        session_id = request.headers.get("X-Saber-Session-Id")
        episode_id = request.headers.get("X-Saber-Episode-Id")

        logger.info(f"🔧 Tools metadata request: session_id={session_id}, episode_id={episode_id}")

        if not tool_registry:
            logger.error("❌ Tools request failed: Tool registry not initialized")
            raise HTTPException(status_code=503, detail="Tool registry not initialized")

        try:
            if not session_id:
                logger.error("❌ Tools request failed: Missing X-Saber-Session-Id header")
                raise HTTPException(status_code=400, detail="Missing X-Saber-Session-Id header")

            # Extract episode and task context
            task_id = request.headers.get("X-Saber-Task-Id")

            # Check if session is registered
            if not session_registry:
                raise HTTPException(status_code=500, detail="Session registry not available")

            agent_id = await session_registry.get_agent_id_by_saber_session(session_id)
            if not agent_id:
                raise HTTPException(status_code=404, detail="Session not registered")

            # Get tools for session+episode using the actual agent_id
            tools = await tool_registry.get_tools_for_session_episode(
                session_id=session_id, episode_id=episode_id, task_id=task_id, agent_id=agent_id
            )

            # Convert to API format
            tools_api = {}
            for tool_name, metadata in tools.items():
                tools_api[tool_name] = {
                    "description": metadata.description,
                    "parameters": metadata.parameters,
                    "timeout": metadata.timeout,
                }

            # Get cached_at timestamp
            cached_at = None
            if tools:
                first_tool = tools[list(tools.keys())[0]]
                if first_tool.cached_at:
                    cached_at = first_tool.cached_at.isoformat()

            tool_count = len(tools)
            logger.info(f"✅ Tools metadata response: {tool_count} tools available for session {session_id}")

            return JSONResponse(
                content={
                    "tools": tools_api,
                    "refresh_interval": tool_registry.cache_ttl.total_seconds(),
                    "cached_at": cached_at,
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Tools request failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/execute_tool")
    async def execute_tool_endpoint(request: Request, body: ToolExecutionRequest) -> JSONResponse:
        """Execute a tool for the requesting session+episode."""
        session_id = request.headers.get("X-Saber-Session-Id")
        episode_id = request.headers.get("X-Saber-Episode-Id")

        logger.info(f"🛠️ Tool execution request: {body.tool_name} for session {session_id}, episode {episode_id}")

        if not tool_registry:
            logger.error("❌ Tool execution failed: Tool registry not initialized")
            raise HTTPException(status_code=503, detail="Tool registry not initialized")

        try:
            if not session_id:
                logger.error("❌ Tool execution failed: Missing X-Saber-Session-Id header")
                raise HTTPException(status_code=400, detail="Missing X-Saber-Session-Id header")

            # Check if session is registered and get agent_id
            if not session_registry:
                raise HTTPException(status_code=500, detail="Session registry not available")

            agent_id = await session_registry.get_agent_id_by_saber_session(session_id)
            if not agent_id:
                raise HTTPException(status_code=404, detail="Session not registered")

            # Extract episode context
            episode_id = request.headers.get("X-Saber-Episode-Id")

            # Emit tool call start event
            import uuid

            call_id = str(uuid.uuid4())
            logger.info(f"� Executing tool {body.tool_name}, call_id={call_id}")

            import time

            start_time = time.time()
            try:
                result = await tool_registry.execute_tool(
                    session_id=session_id,
                    episode_id=episode_id,
                    tool_name=body.tool_name,
                    arguments=body.arguments,
                    timeout=body.timeout,
                    agent_id=agent_id,
                )

                execution_time = time.time() - start_time
                result["execution_time"] = execution_time

                if result.get("success"):
                    logger.info(
                        f"✅ Tool execution successful: {body.tool_name} for session {session_id} "
                        f"({execution_time:.3f}s)"
                    )

                    # Emit successful completion event
                    logger.info(
                        "🚨 Emitting tool_call_complete (success) for %s, call_id=%s",
                        body.tool_name,
                        call_id,
                    )
                    logger.info(f"🔧 Tool execution completed successfully: {body.tool_name}, call_id={call_id}")
                else:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    logger.info(f"� Tool execution failed: {body.tool_name}, call_id={call_id}, error={error_msg}")

                return JSONResponse(content=result)

            except Exception as tool_error:
                execution_time = time.time() - start_time
                logger.error(f"🔧 Tool execution exception: {body.tool_name}, call_id={call_id}, error={tool_error}")
                raise

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {body.tool_name} for session {session_id} - {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/tools/refresh")
    async def refresh_tools(request: Request) -> JSONResponse:
        """Force refresh of tools for the requesting session+episode."""
        if not tool_registry:
            raise HTTPException(status_code=503, detail="Tool registry not initialized")

        try:
            session_id = request.headers.get("X-Saber-Session-Id")
            if not session_id:
                raise HTTPException(status_code=400, detail="Missing X-Saber-Session-Id header")

            # Check if session is registered and get agent_id
            if not session_registry:
                raise HTTPException(status_code=500, detail="Session registry not available")

            agent_id = await session_registry.get_agent_id_by_saber_session(session_id)
            if not agent_id:
                raise HTTPException(status_code=404, detail="Session not registered")

            # Extract episode context
            episode_id = request.headers.get("X-Saber-Episode-Id")
            task_id = request.headers.get("X-Saber-Task-Id")

            # Force refresh
            tools = await tool_registry.get_tools_for_session_episode(
                session_id=session_id, episode_id=episode_id, task_id=task_id, agent_id=agent_id, force_refresh=True
            )

            return JSONResponse(content={"refreshed": True, "tool_count": len(tools), "tools": list(tools.keys())})

        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


# CLI entry point for development/testing
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="warning",  # Only show warnings/errors from uvicorn
        access_log=False,  # Disable access logging
        log_config=None,  # Use our custom logging config
    )


# Create default app instance for imports
app = create_app()


def start_sidecar_service(host: str = "0.0.0.0", port: int = 8002, saber_mcp_url: Optional[str] = None) -> None:
    """Start the MCP sidecar service."""
    import uvicorn

    service_app = create_app(saber_mcp_url=saber_mcp_url)

    # Configure uvicorn with minimal logging to let our Python logging take over
    uvicorn.run(
        service_app,
        host=host,
        port=port,
        log_level="warning",  # Only show warnings/errors from uvicorn
        access_log=False,  # Disable access logging
        log_config=None,  # Use our custom logging config
    )
