"""
Base FastMCP Server for Security-Related Tasks

This module provides a base class for creating FastMCP servers for various security-related tasks.
It abstracts common patterns like context management, environment interaction, evaluation, and
standard MCP tool definitions based on the Excytin benchmark server pattern.
"""

from fastmcp import FastMCP
from typing import Dict, List, Tuple, Union, Any, Optional
import json
import logging


class BaseSecurityTaskServerContext:
    """
    Base class for security task server contexts.
    
    This class provides common attributes and methods that are shared across different security tasks.
    Subclasses should override _initialize_task_environment() to set up their specific environments.
    """

    def __init__(self,
                 attack: Union[str, int],
                 evaluator: Any,
                 save_env_file: Union[str, bool],
                 max_steps: int,
                 split: str,
                 use_full_db: bool,
                 layer: str,
                 q_idx: int = 0):
        """
        Initialize the base security server context.

        Args:
            attack: Identifier for the specific attack/scenario
            evaluator: Evaluator instance for checking responses
            save_env_file: Path to save environment state or False to disable
            max_steps: Maximum number of steps allowed
            split: Data split to use (train/test/val)
            use_full_db: Whether to use full database
            layer: Layer type for the task
            q_idx: Question index to start from
        """
        # Store parameters that will be used by subclasses
        self.current_attack = attack
        self.evaluator = evaluator
        self.save_env_file = save_env_file
        self.max_steps = max_steps
        self.split = split
        self.use_full_db = use_full_db
        self.layer = layer
        self.current_question_idx = q_idx
        
        # Common state tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.accum_reward = 0.0
        self.accum_success = 0
        self.db_connection = None
        
        # Task-specific environment (to be set by subclasses)
        self.task_env = None
        
        # Initialize task-specific components
        self._initialize_task_environment()

    def _initialize_task_environment(self):
        """Initialize task-specific environment. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_task_environment()")


class BaseSecurityTaskServer:
    """
    Base class for FastMCP security task servers.
    
    This class provides a framework for creating MCP servers for security-related tasks
    following the pattern established by the Excytin benchmark server.
    """

    def __init__(self, server_name: str):
        """
        Initialize the base security task server.

        Args:
            server_name: Name of the MCP server
        """
        self.mcp = FastMCP(server_name)
        # Global context instance (following the excytin pattern)
        self._server_context: Optional[BaseSecurityTaskServerContext] = None
        self._setup_resources()
        self._setup_tools()

    def _setup_resources(self):
        """Setup MCP resources."""
        @self.mcp.resource("session://context")
        def get_context() -> str:
            """Get the current server context information."""
            if self._server_context is None:
                return "No context initialized"

            context_info = {
                "current_attack": self._server_context.current_attack,
                "current_question_idx": self._server_context.current_question_idx,
                "step_count": self._server_context.step_count,
                "episode_reward": self._server_context.episode_reward,
                "accum_reward": self._server_context.accum_reward,
                "accum_success": self._server_context.accum_success,
            }
            
            # Add environment-specific info if task_env exists
            if self._server_context.task_env:
                context_info.update({
                    "env_step_count": getattr(self._server_context.task_env, 'step_count', 0),
                    "env_max_steps": getattr(self._server_context.task_env, 'max_steps', 0),
                    "env_current_question": self._get_current_question_text(),
                    "layer": getattr(self._server_context.task_env, 'layer', 'unknown'),
                    "attack_info": getattr(self._server_context.task_env, 'attack', 'unknown')
                })

            return json.dumps(context_info, indent=2)

    def _get_current_question_text(self):
        """Helper to get current question text."""
        if hasattr(self._server_context.task_env, 'curr_question') and self._server_context.task_env.curr_question:
            return self._server_context.task_env.curr_question.get('question', None)
        return None

    def _get_server_context(self) -> BaseSecurityTaskServerContext:
        """Get or create the server context."""
        if self._server_context is None:
            raise ValueError("Server context not initialized. Call initialize_context first.")
        return self._server_context

    def _setup_tools(self):
        """Setup MCP tools following the excytin pattern."""
        
        @self.mcp.tool
        def initialize_context(
            attack: Union[str, int],
            save_env_file: Union[str, bool] = False,
            max_steps: int = 100,
            split: str = "test",
            use_full_db: bool = True,
            layer: str = "alert",
            q_idx: int = 0,
            eval_type: str = "llm",
            evaluator_config: Optional[str] = None
        ) -> str:
            """Initialize the server context with the specified parameters."""
            try:
                # Parse evaluator_config if provided as JSON string
                parsed_evaluator_config = None
                if evaluator_config:
                    try:
                        parsed_evaluator_config = json.loads(evaluator_config)
                    except json.JSONDecodeError:
                        return f"Error: Invalid JSON in evaluator_config: {evaluator_config}"

                # Create evaluator
                evaluator = self._create_evaluator(eval_type, parsed_evaluator_config)
                
                # Handle save file default
                if not save_env_file:
                    save_env_file = f"results/env_{attack}.json"

                # Create context
                self._server_context = self._create_context(
                    attack=attack,
                    evaluator=evaluator,
                    save_env_file=save_env_file,
                    max_steps=max_steps,
                    split=split,
                    use_full_db=use_full_db,
                    layer=layer,
                    q_idx=q_idx
                )

                return f"Context initialized for attack: {attack}, question index: {q_idx}"
            
            except Exception as e:
                return f"Error initializing context: {str(e)}"

        @self.mcp.tool
        def query_sql_database(query: str) -> dict:
            """Query the SQL database with the provided query string."""
            context = self._get_server_context()
            
            print(f"Executing SQL query: {query}")

            try:
                # Use the environment's step method (without submit) to maintain proper state
                observation, reward, done, info = context.task_env.step(query, submit=False)

                # Update context step count
                context.step_count = context.task_env.step_count

                return {
                    "query": query,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "step_count": context.step_count,
                    "status": "success"
                }

            except Exception as e:
                print(f"Error executing SQL query: {e}")
                return {
                    "query": query,
                    "error": str(e),
                    "status": "error"
                }

        @self.mcp.tool
        def submit_answer(answer: str) -> dict:
            """Submit an answer to the current task and receive feedback."""
            context = self._get_server_context()

            print(f"Submitting answer: {answer}")

            try:
                # Use the environment's step method with submit=True
                observation, reward, done, info = context.task_env.step(answer, submit=True)

                return {
                    "answer": answer,
                    "observation": observation.tolist() if hasattr(observation, 'tolist') else str(observation),
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "status": "success"
                }

            except Exception as e:
                print(f"Error submitting answer: {e}")
                return {
                    "answer": answer,
                    "error": str(e),
                    "status": "error"
                }

        @self.mcp.tool
        def get_current_question() -> dict:
            """Get the current question from the environment."""
            context = self._get_server_context()

            try:
                # Get all questions and return the current one
                all_questions = context.task_env.getAllQuestions()
                current_idx = getattr(context.task_env, 'question_idx', 0)

                if current_idx < len(all_questions):
                    current_question = all_questions[current_idx]
                else:
                    current_question = "No more questions available"

                return {
                    "question": current_question,
                    "question_idx": current_idx,
                    "total_questions": len(all_questions),
                    "attack": context.current_attack,
                    "status": "success"
                }

            except Exception as e:
                print(f"Error getting current question: {e}")
                return {
                    "error": str(e),
                    "status": "error"
                }

        @self.mcp.tool
        def reset_environment(q_idx: int = 0) -> dict:
            """Reset the environment to a specific question index."""
            context = self._get_server_context()

            try:
                # Reset the environment
                observation, info = context.task_env.reset(q_idx)
                context.current_question_idx = q_idx

                return {
                    "message": f"Environment reset to question index {q_idx}",
                    "observation": observation,
                    "info": info,
                    "question_idx": q_idx,
                    "status": "success"
                }

            except Exception as e:
                print(f"Error resetting environment: {e}")
                return {
                    "error": str(e),
                    "status": "error"
                }

        @self.mcp.tool
        def get_database_schema(table_name: Optional[str] = None) -> dict:
            """Get the database schema for a specific table or all tables."""
            context = self._get_server_context()

            try:
                if table_name:
                    # Get schema for specific table
                    schema = context.task_env.get_schema(table_name)
                    return {
                        "table_name": table_name,
                        "schema": schema,
                        "status": "success"
                    }
                else:
                    # Get all table names
                    table_names = context.task_env.get_table_names()
                    return {
                        "table_names": table_names,
                        "status": "success"
                    }

            except Exception as e:
                print(f"Error getting database schema: {e}")
                return {
                    "error": str(e),
                    "status": "error"
                }

    def _create_context(self, **kwargs) -> BaseSecurityTaskServerContext:
        """Create a task-specific context. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_context()")

    def _create_evaluator(self, eval_type: str, evaluator_config: Optional[dict]):
        """Create evaluator based on type and config. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_evaluator()")

    def run(self):
        """Run the MCP server."""
        self.mcp.run()
