import asyncio
import json
import re
from typing import Dict, List, Tuple, Union
from fastmcp import Client
from autogen import OpenAIWrapper
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from utils.agent_utils import msging, call_llm, call_llm_foundry, update_model_usage
from config.llm_config import CONFIG_LIST, filter_config_list
import logfire

# Configure logfire for this module
logfire.configure()

BASE_PROMPT = """You are a security analyst. 
You need to answer a given security question by querying the database.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

Your response should always be a thought-action pair:
Thought: <your reasoning>
Action: <your SQL query>

In Thought, you can analyse and reason about the current situation, 
Action can be one of the following: 
(1) execute[<your query>], which executes the SQL query
(2) submit[<your answer>], which is the final answer to the question
"""

BASE_SUMMARY_PROMPT = """You are a security analyst. 
You need to answer a given security question by querying the database.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

Your response should always be a thought-action pair:
Thought: <your reasoning>
Action: <your SQL query>

In Thought, you can analyse and reason about the current situation, 
Action can be one of the following: 
(1) execute[<your query>], which executes the SQL query
(2) submit[<your answer>], which is the final answer to the question

When submitting an answer, please summarize key information from intermediate steps that lead to your answer.
"""

O1_PROMPT = """You are a security analyst. 
You need to answer a given security question by querying the database.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

Your response should always be a thought-action pair:
Thought: <your reasoning>
Action: <your action>

In Thought, you can analyse and reason about the current situation, 
Action can be one of the following: 
(1) execute[<your sql query>], which executes the SQL query. For example, execute[DESCRIBE table_name].
(2) submit[<your answer>], which is the final answer to the question

You should only give one thought-action per response. The action from your response will be executed and the result will be shown to you.
Follow the format "Thought: ....\nAction: ...." exactly.
Do not include any other information in your response. Wait for the response from one action before giving the next thought-action pair. DO NOT make assumptions about the data that are not observed in the logs.
"""

R1_PROMPT = """You are a security analyst. 
You need to interact with a database with MYSQL queries to collect info and then answer a given security question.
The logs are stored in a MySQL database, you can use SQL queries to retrieve entries as needed.
Note there are more than 20 tables in the database, so you may need to explore the schema or check example entries to understand the database structure.

In you response, you should give your thoughts and actions. The reasoning process and action are enclosed within <think> </think> and <answer> </answer> tags,
You should ONLY have ONE action per response in the <answer> </answer> block, it can be one of the following based on your reasoning:
(1) <answer>execute[<your sql query>]</answer>, which executes the SQL query. For example, execute[DESCRIBE table_name]. You should give sql queries to explore the schema and acquire information.
(2) <answer>submit[<your answer>]</answer>, which submits the final answer to the question. When you believe you have enough information to answer the question, you can submit your answer.

Please do not do excessive reasoning. Briefly reason about the current situation and then give your action quickly. DO NOT make assumptions about the data that are not observed in the logs. Be conside and precice in your response.
"""


class MCPBaselineAgent:
    """Baseline agent that interacts with the Excytin Bench MCP server."""
    
    def __init__(self,
                 server_path: str,
                 config_list: List[Dict],
                 cache_seed: int = 41,
                 max_steps: int = 15,
                 submit_summary: bool = False,
                 temperature: float = 0,
                 retry_num: int = 10,
                 retry_wait_time: int = 5,
                 ):
        """
        Initialize the MCP Baseline Agent.
        
        Args:
            server_path: Path to the MCP server script
            config_list: LLM configuration list
            cache_seed: Cache seed for reproducibility
            max_steps: Maximum number of steps
            submit_summary: Whether to submit summary at the end
            temperature: LLM temperature
            retry_num: Number of retries for LLM calls
            retry_wait_time: Wait time between retries
        """
        self.server_path = server_path
        self.cache_seed = cache_seed
        self.config_list = config_list
        self.temperature = temperature
        self.max_steps = max_steps
        self.submit_summary = submit_summary
        self.retry_num = retry_num
        self.retry_wait_time = retry_wait_time
        self.step_count = 0
        self.total_usage = {}
        
        # Initialize MCP client
        self.mcp_client = Client(server_path)
        
        # Adjust temperature for specific models
        if "o4" in config_list[0]['model']:
            self.temperature = 1
        
        # Initialize LLM client
        if "ai_foundry" in config_list[0].get('api_type', ''):
            self.llm_client = ChatCompletionsClient(
                endpoint=config_list[0]['endpoint'],
                credential=AzureKeyCredential(config_list[0]['api_key']),
                seed=self.cache_seed
            )
        else:
            self.llm_client = OpenAIWrapper(config_list=config_list, cache_seed=cache_seed)
        
        # Set system prompt based on model type
        sys_prompt = BASE_SUMMARY_PROMPT if submit_summary else BASE_PROMPT
        if any(model_type in config_list[0]['model'] for model_type in ["o1", "o3", "o4", "meta-llama"]):
            sys_prompt = O1_PROMPT
        elif any(model_type in config_list[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            sys_prompt = R1_PROMPT
            
        self.messages = [{"role": "system", "content": sys_prompt}]
        
        # No system prompt for some models
        if any(model_type in config_list[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            self.messages = [{"role": "system", "content": sys_prompt}]
            print("Using R1/DeepSeek-style prompt")
            logfire.info("Using R1/DeepSeek-style prompt", model=config_list[0]['model'])
    
    @property
    def name(self):
        return "MCPBaselineAgent"
    
    def _call_llm(self, messages):
        """Call the LLM with the given messages."""
        if "ai_foundry" in self.config_list[0].get('api_type', ''):
            response = call_llm_foundry(
                client=self.llm_client,
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature,
                stop=['</answer>'],
            )
            update_model_usage(self.total_usage, model_name=response.model, 
                             usage_dict=response.usage.as_dict())
        else:
            response = call_llm(
                client=self.llm_client,
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature
            )
            update_model_usage(self.total_usage, model_name=response.model, 
                             usage_dict=response.usage.model_dump())
        return response.choices[0].message.content
    
    def _parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse the action string to extract SQL query or answer and determine if it's a submission."""
        # Remove backticks
        action = action.replace("`", "")
        
        # Check for submit action
        if "submit[" in action:
            pattern = r'submit\[(.*)\]'
            matches = re.findall(pattern, action, re.DOTALL)
            if matches:
                return matches[0], True
            return action, True
        
        # Check for execute action
        if "execute[" in action:
            pattern = r'execute\[(.*)\]'
            matches = re.findall(pattern, action, re.DOTALL)
            if matches:
                query = matches[0]
                # Remove semicolon if present
                if ";" in query:
                    query = query[:query.index(";")]
                return query, False
            
        return action, False
    
    async def _initialize_session(self, attack: str, q_idx: int = 0, max_steps: int = 100,
                                  split: str = "test", use_full_db: bool = False, layer: str = "alert"):
        """Initialize the MCP session."""
        try:
            result = await self.mcp_client.call_tool("initialize_context", {
                "attack": attack,
                "max_steps": max_steps,
                "split": split,
                "use_full_db": use_full_db,
                "layer": layer,
                "q_idx": q_idx
            })
            print(f"Session initialized: {result.data if hasattr(result, 'data') else result}")
            logfire.info("Session initialized successfully", 
                        attack=attack, q_idx=q_idx, max_steps=max_steps, 
                        split=split, use_full_db=use_full_db, layer=layer,
                        result=str(result.data if hasattr(result, 'data') else result))
            return True
        except Exception as e:
            print(f"Failed to initialize session: {e}")
            logfire.error("Failed to initialize session", 
                         attack=attack, q_idx=q_idx, error=str(e))
            return False
    
    async def _get_current_question(self):
        """Get the current question from the MCP server."""
        try:
            result = await self.mcp_client.call_tool("get_current_question", {})
            result_data = result.data if hasattr(result, 'data') else result
            logfire.info("Retrieved current question", question=result_data.get('question', {}))
            return result_data.get('question', {})
        except Exception as e:
            print(f"Failed to get current question: {e}")
            logfire.error("Failed to get current question", error=str(e))
            return None
    
    async def _execute_query(self, query: str):
        """Execute a SQL query via the MCP server."""
        try:
            result = await self.mcp_client.call_tool("query_sql_database", {
                "query": query
            })
            result_data = result.data if hasattr(result, 'data') else result
            observation = result_data.get('observation', f"Query executed: {query}")
            logfire.info("SQL query executed", query=query, observation=observation)
            return observation
        except Exception as e:
            print(f"Failed to execute query: {e}")
            logfire.error("Failed to execute SQL query", query=query, error=str(e))
            return f"Error executing query: {str(e)}"
    
    async def _submit_answer(self, answer: str):
        """Submit an answer via the MCP server."""
        try:
            result = await self.mcp_client.call_tool("submit_answer", {
                "answer": answer
            })
            result_data = result.data if hasattr(result, 'data') else result
            logfire.info("Answer submitted", answer=answer, result=result_data)
            return result_data
        except Exception as e:
            print(f"Failed to submit answer: {e}")
            logfire.error("Failed to submit answer", answer=answer, error=str(e))
            return {"error": str(e)}
    
    async def act(self, observation: str = None):
        """
        Generate an action based on the current observation.
        
        Args:
            observation: Current observation (can be None for initial step)
            
        Returns:
            Tuple of (action, is_submit)
        """
        # Add observation to messages if provided
        if observation is not None:
            self._add_message(observation, role="user")
        
        # Generate response from LLM
        response = self._call_llm(messages=self.messages)
        
        # Print agent's thinking with clear formatting
        print("\n" + "="*60)
        print(f"AGENT STEP {self.step_count + 1} - THINKING:")
        print("="*60)
        print(response)
        print("="*60)
        
        # Log agent's thinking
        logfire.info("Agent step thinking", 
                    step=self.step_count + 1, 
                    response=response,
                    model=self.config_list[0]['model'])
        
        # Add summary prompt if we're at max steps
        if self.step_count >= self.max_steps - 1 and self.submit_summary:
            summary_prompt = "You have reached maximum number of steps. Please summarize your findings of key information, and submit them."
            self._add_message(summary_prompt, role="system")
        
        # Parse thought and action
        split_str = "\nAction:"
        if any(model_type in self.config_list[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            split_str = "<answer>"
        
        if "**Action:**" in response:
            split_str = "\n**Action:**"
        
        try:
            thought, action = response.strip().split(split_str)
            action = action.replace("<answer>", "").replace("</answer>", "")
            self._add_message(response.strip(), role="assistant")
        except:
            print("\n[RETRY] Splitting action failed, getting action separately:")
            logfire.warning("Action splitting failed, retrying", 
                           original_response=response.strip(),
                           step=self.step_count + 1)
            thought = response.strip()
            action = self._call_llm(self.messages + [msging(f"{thought}\nAction:")])
            print(f"Action: {action}")
            logfire.info("Action retry successful", action=action)
            action = action.strip()
            action = action.replace("<answer>", "").replace("</answer>", "")
            if "Thought" not in thought:
                thought = f"Thought: {thought}"
            self._add_message(f"{thought}\nAction:{action}", role="assistant")
        
        self.step_count += 1
        
        # Parse the action
        parsed_action, is_submit = self._parse_action(action)
        
        return parsed_action, is_submit
    
    async def run_episode(self, attack: str, q_idx: int = 0, max_steps: int = None,
                          split: str = "test", use_full_db: bool = False, layer: str = "alert"):
        """
        Run a complete episode for a given attack and question.
        
        Args:
            attack: Attack identifier
            q_idx: Question index
            max_steps: Maximum steps (uses agent's max_steps if None)
            split: Data split to use
            use_full_db: Whether to use full database
            layer: Layer type
            
        Returns:
            Dictionary with episode results
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        async with self.mcp_client:
            # Initialize session
            if not await self._initialize_session(attack, q_idx, max_steps, split, use_full_db, layer):
                return {"error": "Failed to initialize session"}
            
            # Get the initial question
            question = await self._get_current_question()
            if question is None:
                return {"error": "Failed to get current question"}
            
            print(f"Question: {json.dumps(question, indent=2)}")
            logfire.info("Episode started", 
                        attack=attack, 
                        q_idx=q_idx, 
                        question=question,
                        max_steps=max_steps)
            
            # Extract question text
            if isinstance(question, dict):
                question_text = question.get('question', str(question))
            else:
                question_text = str(question)
            
            # Start with the question as initial observation
            observation = f"Question: {question_text}"
            
            episode_history = []
            final_result = None
            
            while self.step_count < max_steps:
                print(f"\n{'='*60}")
                print(f"STEP {self.step_count + 1}/{max_steps}")
                print(f"{'='*60}")
                print(f"Current observation: {observation}")
                print(f"{'='*60}")
                
                # Log step start
                logfire.info("Episode step started", 
                           step=self.step_count + 1, 
                           max_steps=max_steps, 
                           observation=observation)
                
                # Get action from agent
                action, is_submit = await self.act(observation)
                
                episode_history.append({
                    "step": self.step_count,
                    "action": action,
                    "is_submit": is_submit,
                    "messages": [msg for msg in self.messages[-2:]]  # Store last 2 messages (user + assistant)
                })
                
                print(f"\n[RESULT] Action to execute: {action}")
                print(f"[RESULT] Is submission: {is_submit}")
                
                # Log action execution
                logfire.info("Action generated", 
                           step=self.step_count, 
                           action=action, 
                           is_submit=is_submit)
                
                if is_submit:
                    # Submit answer and get final result
                    final_result = await self._submit_answer(action)
                    print(f"\n{'='*60}")
                    print("FINAL SUBMISSION")
                    print(f"{'='*60}")
                    print(f"Answer: {action}")
                    print(f"Result: {final_result}")
                    print(f"{'='*60}")
                    
                    # Log final submission
                    logfire.info("Final submission completed", 
                               answer=action, 
                               result=final_result,
                               total_steps=self.step_count,
                               attack=attack,
                               q_idx=q_idx)
                    break
                else:
                    # Execute query and get observation
                    observation = await self._execute_query(action)
                    print(f"\n[QUERY RESULT]")
                    print(f"Query: {action}")
                    print(f"Result: {observation}")
                    episode_history[-1]["observation"] = observation
                    
                    # Log query execution
                    logfire.info("Query executed in episode", 
                               query=action, 
                               observation=observation,
                               step=self.step_count)
            
            # Log episode completion
            logfire.info("Episode completed", 
                        attack=attack,
                        question_idx=q_idx, 
                        total_steps=self.step_count,
                        final_result=final_result,
                        usage_summary=self.total_usage)
            
            return {
                "attack": attack,
                "question_idx": q_idx,
                "question": question,
                "episode_history": episode_history,
                "final_result": final_result,
                "total_steps": self.step_count,
                "usage_summary": self.total_usage
            }
    
    def _add_message(self, msg: str, role: str = "user"):
        """Add a message to the conversation history."""
        self.messages.append(msging(msg, role))
    
    def get_logging(self):
        """Get logging information."""
        return {
            "messages": self.messages,
            "usage_summary": self.total_usage,
        }
    
    def print_conversation_history(self):
        """Print the full conversation history for debugging."""
        print("\n" + "="*80)
        print("FULL CONVERSATION HISTORY")
        print("="*80)
        for i, msg in enumerate(self.messages):
            print(f"\n--- Message {i+1} ({msg['role']}) ---")
            print(msg['content'])
        print("="*80)
        
        # Log conversation history
        logfire.info("Conversation history printed", 
                    message_count=len(self.messages),
                    messages=self.messages)
    
    def reset(self, change_seed: bool = True):
        """Reset the agent state."""
        if change_seed:
            self.cache_seed += 1
        
        # Reinitialize LLM client with new seed
        if "ai_foundry" in self.config_list[0].get('api_type', ''):
            self.llm_client = ChatCompletionsClient(
                endpoint=self.config_list[0]['endpoint'],
                credential=AzureKeyCredential(self.config_list[0]['api_key']),
                seed=self.cache_seed
            )
        elif "azure" in self.config_list[0].get('api_type', ''):
            self.llm_client = OpenAIWrapper(config_list=self.config_list, cache_seed=self.cache_seed)
        
        self.step_count = 0
        
        # Reset system prompt
        sys_prompt = BASE_SUMMARY_PROMPT if self.submit_summary else BASE_PROMPT
        if any(model_type in self.config_list[0]['model'] for model_type in ["o1", "o3", "o4", "meta-llama"]):
            sys_prompt = O1_PROMPT
        elif any(model_type in self.config_list[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            sys_prompt = R1_PROMPT
            
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.total_usage = {}
        
        # Log agent reset
        logfire.info("Agent reset completed", 
                    new_seed=self.cache_seed, 
                    model=self.config_list[0]['model'],
                    seed_changed=change_seed)


# Example usage and testing function
async def test_mcp_baseline_agent():
    """Test function for the MCP Baseline Agent."""
    # Example config - you'll need to provide your actual config
    model_name = "gpt-4o"  # Default eval model, can be overridden
    config_list = filter_config_list(CONFIG_LIST, model_name)
    
    # Initialize agent
    agent = MCPBaselineAgent(
        server_path="excytin_bench_server.py",
        config_list=config_list,
        max_steps=10
    )
    
    # Run an episode
    result = await agent.run_episode(
        attack="incident_5",
        q_idx=0,
        max_steps=10,
        use_full_db=False
    )
    
    print("Episode Results:")
    print(json.dumps(result, indent=2, default=str))
    
    # Log test completion
    logfire.info("Test episode completed", 
                attack="incident_5", 
                result=result)


if __name__ == "__main__":
    asyncio.run(test_mcp_baseline_agent())
