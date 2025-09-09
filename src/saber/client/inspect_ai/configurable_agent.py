"""
SABER Configurable Agent - Plugin-based architecture

Clean factory pattern where the factory looks up inspect_ai agent implementations
and composes them with SABER infrastructure.

Following SABER best practices:
- Composition over inheritance
- Clean factory pattern: factory.create_agent("react")
- Registry stores actual inspect_ai agent implementations
- Configurable agent receives agent implementation as parameter
- Fail-fast design with strict validation
- Type-safe implementation
- No backwards compatibility

Architecture:
- AgentImplementationRegistry: Maps names to inspect_ai agent functions
- AgentFactory: factory.create_agent("react") -> configurable_agent(agent_implementation=react)
- Configurable agent: Composes any agent implementation with SABER infrastructure
"""

import logging
from typing import Any, Callable, Dict, List

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.tool import Tool, mcp_server_http

from ...models import HTTPHeaders
from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .agent_registry import register_inspect_ai_agent

logger = logging.getLogger(__name__)


class AgentImplementationRegistry:
    """Registry for inspect_ai agent implementations."""

    _implementations: Dict[str, Callable] = {}

    @classmethod
    def register_implementation(cls, name: str, agent_func: Callable) -> None:
        """Register an inspect_ai agent implementation.

        Args:
            name: Agent name (e.g., "react", "chain", etc.)
            agent_func: inspect_ai agent function (e.g., inspect_ai.agent.react)
        """
        cls._implementations[name] = agent_func
        logger.info(f"Registered agent implementation: {name}")

    @classmethod
    def get_implementation(cls, name: str) -> Callable:
        """Get agent implementation by name.

        Args:
            name: Agent name

        Returns:
            inspect_ai agent function

        Raises:
            ValueError: If implementation not found
        """
        if name not in cls._implementations:
            available = list(cls._implementations.keys())
            raise ValueError(f"Agent implementation '{name}' not found. Available: {available}")

        return cls._implementations[name]

    @classmethod
    def list_implementations(cls) -> List[str]:
        """List available implementation names."""
        return list(cls._implementations.keys())


# Register built-in inspect_ai agent implementations
def _register_builtin_implementations() -> None:
    """Register standard inspect_ai agent implementations."""
    try:
        from inspect_ai.agent import react

        AgentImplementationRegistry.register_implementation("react", react)
    except ImportError:
        logger.warning("Could not import inspect_ai.agent.react")

    # Add other inspect_ai agents as they become available
    # try:
    #     from inspect_ai.agent import chain
    #     AgentImplementationRegistry.register_implementation("chain", chain)
    # except ImportError:
    #     pass


_register_builtin_implementations()


# Register the factory-based agents with the SABER inspect_ai agent registry
@register_inspect_ai_agent(
    name="react",
    description="React agent with native SABER MCP integration",
    capabilities=["reasoning", "tool_use", "step_by_step"],
    tags=["saber", "react", "security"],
)
async def create_react_agent(
    config: SABERConfig,
    session_manager: ClientSessionManager,
    agent_id: str = "react",
    **kwargs: Any,
) -> Agent:
    """Create a React agent with SABER MCP integration."""
    # Delegate to configurable agent with react type
    return await create_configurable_agent(
        config=config,
        session_manager=session_manager,
        agent_id=agent_id,
        agent_type="react",
        **kwargs,
    )


@register_inspect_ai_agent(
    name="configurable",
    description="Configurable SABER agent with pluggable implementations and native MCP",
    capabilities=["reasoning", "tool_use", "configurable_behavior"],
    tags=["saber", "configurable", "security"],
)
async def create_configurable_agent(
    config: SABERConfig,
    session_manager: ClientSessionManager,
    agent_id: str = "configurable",
    agent_type: str = "react",
    **kwargs: Any,
) -> Agent:
    """Create a configurable agent with native MCP integration."""

    # Get the agent implementation
    agent_implementation = AgentImplementationRegistry.get_implementation(agent_type)

    # Create a SABER-aware configurable agent using inspect_ai's @agent decorator
    @agent  # type: ignore[misc]
    def saber_configurable_agent() -> Agent:
        """SABER-aware configurable agent with deferred context creation."""

        async def execute(state: AgentState, tools: List[Tool]) -> AgentState:
            """Agent execution function called for each sample - NOW sample metadata is available."""

            # Import what we need for SABER context initialization
            from inspect_ai.solver._task_state import sample_state
            from inspect_ai.util import store

            # Initialize SABER context similar to saber_react.py
            task_store = store()

            # Store session manager
            task_store.set("saber_session_manager", session_manager)

            if session_manager is None:
                raise ValueError("Session manager is required but not provided")

            session_id = session_manager.get_current_session_id()
            if session_id is None:
                raise ValueError("Session ID not available from session manager")
            task_store.set("saber_session_id", session_id)

            # Get task_id from current sample metadata using sample_state()
            current_state = sample_state()
            if current_state is None:
                raise ValueError("Current task state is not available")

            task_id = current_state.metadata.get("task_id")
            if task_id is None:
                raise ValueError("Task ID not found in sample metadata")
            task_store.set("saber_task_id", task_id)

            # Get initial prompt from sample metadata (server-provided at task creation)
            initial_prompt = current_state.metadata.get("initial_prompt")
            if initial_prompt is None:
                raise ValueError("Initial prompt not found in sample metadata")
            logger.info(f"Using initial prompt from metadata: {initial_prompt[:100]}...")

            # Create episode
            episode = await session_manager.create_episode(session_id, task_id)
            task_store.set("saber_current_episode", episode)

            # Create MCP server connection with episode headers using SABER standard headers
            mcp_headers = {
                HTTPHeaders.SESSION_ID: session_id,
                HTTPHeaders.EPISODE_ID: episode.episode_id,
                HTTPHeaders.TASK_ID: task_id,
            }

            saber_server = mcp_server_http(
                name="SABER Security Tools",
                url=f"{config.saber_mcp_url}/mcp",
                headers=mcp_headers,
            )

            # Combine all tools (passed tools + SABER MCP tools)
            all_tools = list(tools) + [saber_server]

            # Create the actual agent with SABER tools using the specified implementation
            actual_agent = agent_implementation(
                name=f"SABER {agent_id.title()} Agent",
                prompt=initial_prompt,
                tools=all_tools,
                **kwargs,
            )

            try:
                # Run the agent
                result = await actual_agent(state)

                # End episode with success
                await session_manager.end_episode(
                    episode.session_id, episode.episode_id, reason="completed", result="success"
                )

                return result
            except Exception as e:
                # End episode with error
                await session_manager.end_episode(episode.session_id, episode.episode_id, reason="error", result=str(e))
                raise

        return execute

    # Return the SABER-aware configurable agent
    return saber_configurable_agent()
