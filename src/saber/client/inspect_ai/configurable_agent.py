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
from typing import Any, Callable, Dict, List, Optional

from inspect_ai.agent import Agent, AgentState
from inspect_ai.tool import Tool

from ..client_session import ClientSessionManager
from ..models import SABERConfig
from .agent_registry import register_inspect_ai_agent
from .saber_agent_base import SABERAgentContext

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


class SABERAgentFactory:
    """Factory for creating SABER-enhanced agents."""

    def __init__(self, registry: Optional[AgentImplementationRegistry] = None):
        """Initialize factory with implementation registry."""
        self.registry = registry or AgentImplementationRegistry()

    async def create_agent(
        self,
        agent_type: str,
        config: SABERConfig,
        session_manager: ClientSessionManager,
        agent_id: Optional[str] = None,
        **agent_params: Any,
    ) -> Agent:
        """Create a SABER-enhanced agent.

        Args:
            agent_type: Type of agent implementation ("react", "chain", etc.)
            config: SABER configuration
            session_manager: Session manager for SABER infrastructure
            agent_id: Agent identifier (defaults to agent_type)
            **agent_params: Parameters to pass to the agent implementation

        Returns:
            SABER-enhanced inspect_ai Agent
        """
        # Get the inspect_ai agent implementation
        agent_implementation = self.registry.get_implementation(agent_type)

        # Use agent_type as agent_id if not provided
        if agent_id is None:
            agent_id = agent_type

        logger.info(f"Creating SABER agent: {agent_id} using implementation: {agent_type}")

        # Return the configurable agent with the specific implementation
        return await self._create_configurable_agent(
            agent_implementation=agent_implementation,
            agent_id=agent_id,
            session_manager=session_manager,
            agent_params=agent_params,
        )

    async def _create_configurable_agent(
        self,
        agent_implementation: Callable,
        agent_id: str,
        session_manager: ClientSessionManager,
        agent_params: Dict[str, Any],
    ) -> Agent:
        """Create configurable agent with specific implementation."""

        from inspect_ai.agent import agent

        # Create the agent function with proper @agent decoration
        @agent  # type: ignore[misc]
        def saber_configurable_agent() -> Agent:
            """SABER configurable agent factory."""

            async def configurable_agent(state: AgentState, tools: List[Tool]) -> AgentState:
                """Configurable agent that composes any inspect_ai agent with SABER infrastructure."""

                # Initialize SABER context
                async with SABERAgentContext(
                    session_manager=session_manager,
                    agent_id=agent_id,
                    default_prompt=f"You are a {agent_id} security domain agent.",
                ) as context:

                    # Get enhanced state and tools
                    enhanced_state = context.get_enhanced_state(state)
                    all_tools = context.get_all_tools(tools)

                    # Create the specific agent implementation with SABER-enhanced parameters
                    enhanced_params = {
                        "name": f"SABER {agent_id.title()} Agent",
                        "description": f"SABER-enhanced {agent_id} agent for security domain",
                        "prompt": context.enhanced_prompt,
                        "tools": all_tools,
                        **agent_params,  # Add any additional parameters
                    }

                    # Create and execute the agent implementation
                    agent_instance = agent_implementation(**enhanced_params)
                    result = await agent_instance(enhanced_state)

                    logger.info(f"SABER agent ({agent_id}) execution completed")
                    return result

            return configurable_agent

        # Call the decorated factory to get the actual agent
        return saber_configurable_agent()


# Register the factory-based agents with the SABER inspect_ai agent registry
@register_inspect_ai_agent(
    name="react",
    description="ReAct agent with SABER infrastructure",
    capabilities=["reasoning", "tool_use"],
    tags=["saber", "react", "security"],
)
async def create_react_agent(
    config: SABERConfig,
    session_manager: ClientSessionManager,
    agent_id: str = "react",
    attempts: int = 1,
    **kwargs: Any,
) -> Agent:
    """Create a ReAct agent using the factory pattern."""
    factory = SABERAgentFactory()
    return await factory.create_agent(
        agent_type="react",
        config=config,
        session_manager=session_manager,
        agent_id=agent_id,
        attempts=attempts,
        **kwargs,
    )


@register_inspect_ai_agent(
    name="configurable",
    description="Configurable SABER agent with pluggable implementations",
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
    """Create a configurable agent using the factory pattern."""
    factory = SABERAgentFactory()
    return await factory.create_agent(
        agent_type=agent_type, config=config, session_manager=session_manager, agent_id=agent_id, **kwargs
    )
