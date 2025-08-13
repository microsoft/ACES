from fastmcp import FastMCP, Context
from evaluator import LLMEvaluator, StaticEvaluator, Evaluator
import gymnasium as gym
import numpy as np
import datetime, json, logging, os, re
from typing import Dict, List, Tuple, Union
import docker
import mysql.connector
from datetime import datetime
from time import sleep
from config.env_config import ATTACKS, AlphineSkiHouseInfo
from config.llm_config import CONFIG_LIST
from utils.env_utils import start_container
from excytin_env import ExcytinEnv
from utils.utils import filter_config_list
import logfire

# Configure logfire for this module
logfire.configure()

mcp = FastMCP("Excytin Bench Threat Investigation Server")

class ExcytinBenchServerContext:
    """Context for Excytin Bench Server"""

    def __init__(self,
        attack: Union[str, int],
        evaluator: Evaluator,
        save_env_file: Union[str, bool],
        max_steps: int,
        split: str,
        use_full_db: bool,
        layer: str,
        # Pick question index to start from, default is 0
        q_idx: int = 0
    ):

        logfire.info("Initializing ExcytinBenchServerContext", 
                    attack=attack, 
                    q_idx=q_idx, 
                    max_steps=max_steps,
                    split=split, 
                    use_full_db=use_full_db, 
                    layer=layer,
                    evaluator_type=type(evaluator).__name__)

        # Initializing environment
        self.thug_env = ExcytinEnv(
            attack=attack,
            evaluator=evaluator,
            save_file=save_env_file,
            max_steps=max_steps,
            split=split,
            use_full_db=use_full_db,
            layer=layer,
        )

        # Resetting environment to start from the specified question index
        self.thug_env.reset(q_idx)

        # Store database connection info
        self.db_connection = None
        self.current_attack = attack
        self.current_question_idx = q_idx

        #Initialize episode state
        self.step_count = 0
        self.episode_reward = 0.0
        self.accum_reward = 0.0
        self.accum_success = 0
        
        logfire.info("ExcytinBenchServerContext initialized successfully", 
                    attack=attack, 
                    q_idx=q_idx)

# Global context instance
_server_context: ExcytinBenchServerContext = None

@mcp.resource("session://context")
def get_context() -> str:
    """Get the current server context information."""
    logfire.info("MCP resource requested: get_context")
    
    if _server_context is None:
        logfire.warning("Context requested but not initialized")
        return "No context initialized"

    context_info = {
        "current_attack": _server_context.current_attack,
        "current_question_idx": _server_context.current_question_idx,
        "step_count": _server_context.step_count,
        "episode_reward": _server_context.episode_reward,
        "accum_reward": _server_context.accum_reward,
        "accum_success": _server_context.accum_success,
        "env_step_count": getattr(_server_context.thug_env, 'step_count', 0),
        "env_max_steps": getattr(_server_context.thug_env, 'max_steps', 0),
        "env_current_question": getattr(_server_context.thug_env, 'curr_question', None)['question'] if hasattr(_server_context.thug_env, 'curr_question') else None,
        "layer": getattr(_server_context.thug_env, 'layer', 'unknown'),
        "attack_info": getattr(_server_context.thug_env, 'attack', 'unknown')
    }

    logfire.info("Context information retrieved", context_info=context_info)
    return json.dumps(context_info, indent=2)

def get_server_context() -> ExcytinBenchServerContext:
    """Get or create the server context."""
    global _server_context
    if _server_context is None:
        # Initialize with default values - this should be called by the client
        logfire.error("Server context requested but not initialized")
        raise ValueError("Server context not initialized. Call initialize_context first.")
    
    logfire.debug("Server context accessed", 
                 attack=_server_context.current_attack,
                 question_idx=_server_context.current_question_idx,
                 step_count=_server_context.step_count)
    
    return _server_context

@mcp.tool
def initialize_context(
    attack: Union[str, int],
    save_env_file: Union[str, bool] = False,
    max_steps: int = 100,
    split: str = "test",
    use_full_db: bool = True,
    layer: str = "alert",
    q_idx: int = 0,
    eval_type: str = "llm",
    evaluator_config: dict = None
) -> str:
    """Initialize the server context with the specified parameters."""
    global _server_context

    logfire.info("MCP tool called: initialize_context", 
                attack=attack, 
                save_env_file=save_env_file,
                max_steps=max_steps, 
                split=split, 
                use_full_db=use_full_db,
                layer=layer, 
                q_idx=q_idx, 
                eval_type=eval_type,
                evaluator_config=evaluator_config)

    try:
        if eval_type == "llm":
            if evaluator_config:
                evaluator = LLMEvaluator(**evaluator_config)
                logfire.info("Using custom LLM evaluator configuration", config=evaluator_config)
            else:
                #setting defaul eval configs
                eval_model = "gpt-4o"  # Default eval model, can be overridden
                eval_config_list = filter_config_list(CONFIG_LIST, eval_model)
                cache_seed = 100 # Default cache seed, can be overridden
                evaluator = LLMEvaluator(
                    config_list=eval_config_list,
                    cache_seed=cache_seed,
                    ans_check_reflection=True,
                    sol_check_reflection=True,
                    step_checking=True,
                    strict_check=False,
                )
                logfire.info("Using default LLM evaluator configuration", 
                           eval_model=eval_model, 
                           cache_seed=cache_seed)
        elif eval_type == "static":
            if evaluator_config:
                evaluator = StaticEvaluator(**evaluator_config)
                logfire.info("Using custom static evaluator configuration", config=evaluator_config)
            else:
                evaluator = StaticEvaluator()
                logfire.info("Using default static evaluator configuration")
        else:
            logfire.error("Unknown eval_type provided", eval_type=eval_type)
            raise ValueError(f"Unknown eval_type: {eval_type}")

        if not save_env_file:
            #use default save file name and path
            base_dir = "results"
            save_env_file = f"{base_dir}/env_{attack}.json"
            logfire.info("Using default save file path", save_env_file=save_env_file)

        _server_context = ExcytinBenchServerContext(
            attack=attack,
            evaluator = evaluator,
            save_env_file=save_env_file,
            max_steps=max_steps,
            split=split,
            use_full_db=use_full_db,
            layer=layer,
            q_idx=q_idx,
        )

        result_msg = f"Context initialized for attack: {attack}, question index: {q_idx}"
        logfire.info("MCP context initialization completed successfully", 
                    attack=attack, 
                    q_idx=q_idx, 
                    result=result_msg)
        return result_msg
        
    except Exception as e:
        logfire.error("Failed to initialize MCP context", 
                     attack=attack, 
                     error=str(e), 
                     error_type=type(e).__name__)
        raise

@mcp.tool
def query_sql_database(query: str) -> dict:
    """Query the SQL database with the provided query string."""

    logfire.info("MCP tool called: query_sql_database", query=query)

    # Get the server context
    context = get_server_context()

    print(f"Executing SQL query: {query}")

    try:
        # Use the environment's step method (without submit) to maintain proper state
        observation, reward, done, info = context.thug_env.step(query, submit=False)

        # Update context step count
        old_step_count = context.step_count
        context.step_count = context.thug_env.step_count

        result = {
            "query": query,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "step_count": context.step_count,
            "status": "success"
        }

        logfire.info("SQL query executed successfully", 
                    query=query,
                    reward=reward,
                    done=done,
                    step_count=context.step_count,
                    prev_step_count=old_step_count,
                    observation_length=len(str(observation)),
                    info=info)

        return result

    except Exception as e:
        print(f"Error executing SQL query: {e}")
        error_result = {
            "query": query,
            "error": str(e),
            "status": "error"
        }
        
        logfire.error("SQL query execution failed", 
                     query=query, 
                     error=str(e), 
                     error_type=type(e).__name__)
        
        return error_result

@mcp.tool
def submit_answer(answer: str) -> dict:
    """Submit an answer to the current task and receive feedback."""

    logfire.info("MCP tool called: submit_answer", answer=answer)

    # Get the server context
    context = get_server_context()

    print(f"Submitting answer: {answer}")

    try:
        # Use the environment's step method with submit=True
        observation, reward, done, info = context.thug_env.step(answer, submit=True)

        result = {
            "answer": answer,
            "observation": observation.tolist() if hasattr(observation, 'tolist') else str(observation),
            "reward": reward,
            "done": done,
            "info": info,
            "status": "success"
        }

        logfire.info("Answer submitted successfully", 
                    answer=answer,
                    reward=reward,
                    done=done,
                    info=info,
                    attack=context.current_attack,
                    question_idx=context.current_question_idx,
                    step_count=context.step_count)

        return result

    except Exception as e:
        print(f"Error submitting answer: {e}")
        error_result = {
            "answer": answer,
            "error": str(e),
            "status": "error"
        }
        
        logfire.error("Answer submission failed", 
                     answer=answer, 
                     error=str(e), 
                     error_type=type(e).__name__,
                     attack=context.current_attack,
                     question_idx=context.current_question_idx)
        
        return error_result

@mcp.tool
def get_current_question() -> dict:
    """Get the current question from the environment."""

    logfire.info("MCP tool called: get_current_question")

    # Get the server context
    context = get_server_context()

    try:
        # Get all questions and return the current one
        all_questions = context.thug_env.getAllQuestions()
        current_idx = getattr(context.thug_env, 'question_idx', 0)

        if current_idx < len(all_questions):
            current_question = all_questions[current_idx]
        else:
            current_question = "No more questions available"

        result = {
            "question": current_question,
            "question_idx": current_idx,
            "total_questions": len(all_questions),
            "attack": context.current_attack,
            "status": "success"
        }

        logfire.info("Current question retrieved successfully", 
                    question_idx=current_idx,
                    total_questions=len(all_questions),
                    attack=context.current_attack,
                    question=current_question)

        return result

    except Exception as e:
        print(f"Error getting current question: {e}")
        error_result = {
            "error": str(e),
            "status": "error"
        }
        
        logfire.error("Failed to get current question", 
                     error=str(e), 
                     error_type=type(e).__name__,
                     attack=getattr(context, 'current_attack', 'unknown'))
        
        return error_result

@mcp.tool
def reset_environment(q_idx: int = 0) -> dict:
    """Reset the environment to a specific question index."""

    logfire.info("MCP tool called: reset_environment", q_idx=q_idx)

    # Get the server context
    context = get_server_context()

    try:
        old_question_idx = context.current_question_idx
        
        # Reset the environment
        observation, info = context.thug_env.reset(q_idx)
        context.current_question_idx = q_idx

        result = {
            "message": f"Environment reset to question index {q_idx}",
            "observation": observation,
            "info": info,
            "question_idx": q_idx,
            "status": "success"
        }

        logfire.info("Environment reset successfully", 
                    old_question_idx=old_question_idx,
                    new_question_idx=q_idx,
                    attack=context.current_attack,
                    info=info)

        return result

    except Exception as e:
        print(f"Error resetting environment: {e}")
        error_result = {
            "error": str(e),
            "status": "error"
        }
        
        logfire.error("Failed to reset environment", 
                     q_idx=q_idx,
                     error=str(e), 
                     error_type=type(e).__name__,
                     attack=getattr(context, 'current_attack', 'unknown'))
        
        return error_result

@mcp.tool
def get_database_schema(table_name: str = None) -> dict:
    """Get the database schema for a specific table or all tables."""

    logfire.info("MCP tool called: get_database_schema", table_name=table_name)

    # Get the server context
    context = get_server_context()

    try:
        if table_name:
            # Get schema for specific table
            schema = context.thug_env.get_schema(table_name)
            result = {
                "table_name": table_name,
                "schema": schema,
                "status": "success"
            }
            
            logfire.info("Database schema retrieved for specific table", 
                        table_name=table_name,
                        schema_fields=len(schema) if isinstance(schema, (list, dict)) else 'unknown',
                        attack=context.current_attack)
        else:
            # Get all table names
            table_names = context.thug_env.get_table_names()
            result = {
                "table_names": table_names,
                "status": "success"
            }
            
            logfire.info("Database table names retrieved", 
                        table_count=len(table_names) if isinstance(table_names, list) else 'unknown',
                        table_names=table_names,
                        attack=context.current_attack)

        return result

    except Exception as e:
        print(f"Error getting database schema: {e}")
        error_result = {
            "error": str(e),
            "status": "error"
        }
        
        logfire.error("Failed to get database schema", 
                     table_name=table_name,
                     error=str(e), 
                     error_type=type(e).__name__,
                     attack=getattr(context, 'current_attack', 'unknown'))
        
        return error_result

if __name__ == "__main__":
    logfire.info("Starting Excytin Bench MCP Server")
    try:
        mcp.run()
    except Exception as e:
        logfire.error("MCP Server startup failed", 
                     error=str(e), 
                     error_type=type(e).__name__)
        raise
    finally:
        logfire.info("Excytin Bench MCP Server shutting down")
