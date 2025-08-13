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


class MCPMultiModelBaselineAgent:
    """Multi-model baseline agent that switches between master and slave models for the Excytin Bench MCP server."""
    
    def __init__(self,
                 server_path: str,
                 config_list_master: List[Dict],
                 config_list_slave: List[Dict],
                 cache_seed: int = 41,
                 max_steps: int = 15,
                 submit_summary: bool = False,
                 temperature: float = 0,
                 retry_num: int = 10,
                 retry_wait_time: int = 5,
                 switch_interval: int = 5,
                 ):
        """
        Initialize the MCP Multi-Model Baseline Agent.
        
        Args:
            server_path: Path to the MCP server script
            config_list_master: Master LLM configuration list (stronger model)
            config_list_slave: Slave LLM configuration list (weaker/faster model)
            cache_seed: Cache seed for reproducibility
            max_steps: Maximum number of steps
            submit_summary: Whether to submit summary at the end
            temperature: LLM temperature
            retry_num: Number of retries for LLM calls
            retry_wait_time: Wait time between retries
            switch_interval: Switch to master model every N steps
        """
        self.server_path = server_path
        self.cache_seed = cache_seed
        self.config_list_master = config_list_master
        self.config_list_slave = config_list_slave
        self.config_list = config_list_slave  # Start with slave model
        self.temperature = temperature
        self.max_steps = max_steps
        self.submit_summary = submit_summary
        self.retry_num = retry_num
        self.retry_wait_time = retry_wait_time
        self.step_count = 0
        self.total_usage = {}
        self.master_usage = {}
        self.slave_usage = {}
        self.switch_interval = switch_interval
        
        # Initialize MCP client
        self.mcp_client = Client(server_path)
        
        # Adjust temperature for specific models
        if "o4" in config_list_master[0]['model'] or "o4" in config_list_slave[0]['model']:
            self.temperature = 1
        
        # Initialize LLM clients
        if "ai_foundry" in config_list_master[0].get('api_type', ''):
            self.master_client = ChatCompletionsClient(
                endpoint=config_list_master[0]['endpoint'],
                credential=AzureKeyCredential(config_list_master[0]['api_key']),
                seed=self.cache_seed
            )
        else:
            self.master_client = OpenAIWrapper(config_list=config_list_master, cache_seed=cache_seed)
            
        if "ai_foundry" in config_list_slave[0].get('api_type', ''):
            self.slave_client = ChatCompletionsClient(
                endpoint=config_list_slave[0]['endpoint'],
                credential=AzureKeyCredential(config_list_slave[0]['api_key']),
                seed=self.cache_seed
            )
        else:
            self.slave_client = OpenAIWrapper(config_list=config_list_slave, cache_seed=cache_seed)
        
        # Start with slave client
        self.current_client = self.slave_client
        self.current_usage = self.slave_usage
        
        # Set system prompt based on model type
        sys_prompt = BASE_SUMMARY_PROMPT if submit_summary else BASE_PROMPT
        if any(model_type in config_list_slave[0]['model'] for model_type in ["o1", "o3", "o4", "meta-llama"]):
            sys_prompt = O1_PROMPT
        elif any(model_type in config_list_slave[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            sys_prompt = R1_PROMPT
            
        self.messages = [{"role": "system", "content": sys_prompt}]
        
        # No system prompt for some models
        if any(model_type in config_list_slave[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            self.messages = [{"role": "system", "content": sys_prompt}]
            print("Using R1/DeepSeek-style prompt with multi-model switching")
    
    @property
    def name(self):
        return "MCPMultiModelBaselineAgent"
    
    def _switch_model(self):
        """Switch between master and slave models based on step count."""
        # Switch to master model every switch_interval steps (but not at step 0)
        if (self.step_count % self.switch_interval == 0) and (self.step_count != 0):
            self.current_client = self.master_client
            self.config_list = self.config_list_master
            self.current_usage = self.master_usage
            model_name = "MASTER"
        else:
            self.current_client = self.slave_client
            self.config_list = self.config_list_slave
            self.current_usage = self.slave_usage
            model_name = "SLAVE"
        
        print(f"[MODEL SWITCH] Using {model_name} model: {self.config_list[0]['model']}")
        logfire.info("Multi-model agent switching models", 
                    model_type=model_name,
                    model_name=self.config_list[0]['model'],
                    step=self.step_count,
                    switch_interval=self.switch_interval)
    
    def _call_llm(self, messages):
        """Call the current LLM with the given messages."""
        if "ai_foundry" in self.config_list[0].get('api_type', ''):
            response = call_llm_foundry(
                client=self.current_client,
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature,
                stop=['</answer>'],
            )
            update_model_usage(self.current_usage, model_name=response.model, 
                             usage_dict=response.usage.as_dict())
            update_model_usage(self.total_usage, model_name=response.model, 
                             usage_dict=response.usage.as_dict())
        else:
            response = call_llm(
                client=self.current_client,
                model=self.config_list[0]['model'],
                messages=messages,
                retry_num=self.retry_num,
                retry_wait_time=self.retry_wait_time,
                temperature=self.temperature
            )
            update_model_usage(self.current_usage, model_name=response.model, 
                             usage_dict=response.usage.model_dump())
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
            return True
        except Exception as e:
            print(f"Failed to initialize session: {e}")
            return False
    
    async def _get_current_question(self):
        """Get the current question from the MCP server."""
        try:
            result = await self.mcp_client.call_tool("get_current_question", {})
            result_data = result.data if hasattr(result, 'data') else result
            return result_data.get('question', {})
        except Exception as e:
            print(f"Failed to get current question: {e}")
            return None
    
    async def _execute_query(self, query: str):
        """Execute a SQL query via the MCP server."""
        try:
            result = await self.mcp_client.call_tool("query_sql_database", {
                "query": query
            })
            result_data = result.data if hasattr(result, 'data') else result
            return result_data.get('observation', f"Query executed: {query}")
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return f"Error executing query: {str(e)}"
    
    async def _submit_answer(self, answer: str):
        """Submit an answer via the MCP server."""
        try:
            result = await self.mcp_client.call_tool("submit_answer", {
                "answer": answer
            })
            result_data = result.data if hasattr(result, 'data') else result
            return result_data
        except Exception as e:
            print(f"Failed to submit answer: {e}")
            return {"error": str(e)}
    
    async def act(self, observation: str = None):
        """
        Generate an action based on the current observation.
        
        Args:
            observation: Current observation (can be None for initial step)
            
        Returns:
            Tuple of (action, is_submit)
        """
        # Switch model based on step count
        self._switch_model()
        
        # Add observation to messages if provided
        if observation is not None:
            self._add_message(observation, role="user")
        
        # Generate response from LLM
        response = self._call_llm(messages=self.messages)
        
        # Print agent's thinking with clear formatting
        current_model = "MASTER" if self.current_client == self.master_client else "SLAVE"
        print("\n" + "="*60)
        print(f"MULTI-MODEL AGENT STEP {self.step_count + 1} - THINKING ({current_model}):")
        print(f"Model: {self.config_list[0]['model']}")
        print("="*60)
        print(response)
        print("="*60)
        
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
            thought = response.strip()
            action = self._call_llm(self.messages + [msging(f"{thought}\nAction:")])
            print(f"Action: {action}")
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
                print(f"MULTI-MODEL STEP {self.step_count + 1}/{max_steps}")
                print(f"{'='*60}")
                print(f"Current observation: {observation}")
                print(f"{'='*60}")
                
                # Get action from agent
                action, is_submit = await self.act(observation)
                
                episode_history.append({
                    "step": self.step_count,
                    "action": action,
                    "is_submit": is_submit,
                    "model_used": "master" if self.current_client == self.master_client else "slave",
                    "model_name": self.config_list[0]['model'],
                    "messages": [msg for msg in self.messages[-2:]]  # Store last 2 messages (user + assistant)
                })
                
                print(f"\n[RESULT] Action to execute: {action}")
                print(f"[RESULT] Is submission: {is_submit}")
                
                if is_submit:
                    # Submit answer and get final result
                    final_result = await self._submit_answer(action)
                    print(f"\n{'='*60}")
                    print("FINAL SUBMISSION")
                    print(f"{'='*60}")
                    print(f"Answer: {action}")
                    print(f"Result: {final_result}")
                    print(f"{'='*60}")
                    break
                else:
                    # Execute query and get observation
                    observation = await self._execute_query(action)
                    print(f"\n[QUERY RESULT]")
                    print(f"Query: {action}")
                    print(f"Result: {observation}")
                    episode_history[-1]["observation"] = observation
            
            return {
                "attack": attack,
                "question_idx": q_idx,
                "question": question,
                "episode_history": episode_history,
                "final_result": final_result,
                "total_steps": self.step_count,
                "usage_summary": self.total_usage,
                "master_usage": self.master_usage,
                "slave_usage": self.slave_usage,
                "switch_interval": self.switch_interval
            }
    
    def _add_message(self, msg: str, role: str = "user"):
        """Add a message to the conversation history."""
        self.messages.append(msging(msg, role))
    
    def get_logging(self):
        """Get logging information."""
        return {
            "messages": self.messages,
            "usage_summary": self.total_usage,
            "master_usage": self.master_usage,
            "slave_usage": self.slave_usage,
        }
    
    def print_conversation_history(self):
        """Print the full conversation history for debugging."""
        print("\n" + "="*80)
        print("MULTI-MODEL AGENT CONVERSATION HISTORY")
        print("="*80)
        for i, msg in enumerate(self.messages):
            print(f"\n--- Message {i+1} ({msg['role']}) ---")
            print(msg['content'])
        print("="*80)
    
    def reset(self, change_seed: bool = True):
        """Reset the agent state."""
        if change_seed:
            self.cache_seed += 1
        
        # Reinitialize LLM clients with new seed
        if "ai_foundry" in self.config_list_master[0].get('api_type', ''):
            self.master_client = ChatCompletionsClient(
                endpoint=self.config_list_master[0]['endpoint'],
                credential=AzureKeyCredential(self.config_list_master[0]['api_key']),
                seed=self.cache_seed
            )
        elif "azure" in self.config_list_master[0].get('api_type', ''):
            self.master_client = OpenAIWrapper(config_list=self.config_list_master, cache_seed=self.cache_seed)
            
        if "ai_foundry" in self.config_list_slave[0].get('api_type', ''):
            self.slave_client = ChatCompletionsClient(
                endpoint=self.config_list_slave[0]['endpoint'],
                credential=AzureKeyCredential(self.config_list_slave[0]['api_key']),
                seed=self.cache_seed
            )
        elif "azure" in self.config_list_slave[0].get('api_type', ''):
            self.slave_client = OpenAIWrapper(config_list=self.config_list_slave, cache_seed=self.cache_seed)
        
        self.step_count = 0
        self.config_list = self.config_list_slave  # Start with slave
        self.current_client = self.slave_client
        self.current_usage = self.slave_usage
        
        # Reset system prompt
        sys_prompt = BASE_SUMMARY_PROMPT if self.submit_summary else BASE_PROMPT
        if any(model_type in self.config_list_slave[0]['model'] for model_type in ["o1", "o3", "o4", "meta-llama"]):
            sys_prompt = O1_PROMPT
        elif any(model_type in self.config_list_slave[0]['model'] for model_type in ["r1", "R1", "qwen3"]):
            sys_prompt = R1_PROMPT
            
        self.messages = [{"role": "system", "content": sys_prompt}]
        self.total_usage = {}
        self.master_usage = {}
        self.slave_usage = {}


# Example usage and testing function
async def test_mcp_multi_model_baseline_agent():
    """Test function for the MCP Multi-Model Baseline Agent."""
    # Example config - you'll need to provide your actual config
    model_name_master = "gpt-4o"  # Strong model
    model_name_slave = "gpt-4o-mini"  # Fast/cheap model
    config_list_master = filter_config_list(CONFIG_LIST, model_name_master)
    config_list_slave = filter_config_list(CONFIG_LIST, model_name_slave)
    
    # Initialize agent
    agent = MCPMultiModelBaselineAgent(
        server_path="excytin_bench_server.py",
        config_list_master=config_list_master,
        config_list_slave=config_list_slave,
        max_steps=10,
        switch_interval=3
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


if __name__ == "__main__":
    asyncio.run(test_mcp_multi_model_baseline_agent())
