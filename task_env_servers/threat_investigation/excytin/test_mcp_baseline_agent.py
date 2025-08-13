#!/usr/bin/env python3
"""
Test script for the MCP Baseline Agent.
This script demonstrates how to use the MCPBaselineAgent to interact with the Excytin Bench MCP server.
"""

import asyncio
import json
import sys
import os
from typing import Dict, List
from config.llm_config import CONFIG_LIST, filter_config_list

# Add the current directory to the path to find the agents module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

model_name = "gpt-4o"  # Default eval model, can be overridden
config_list = filter_config_list(CONFIG_LIST, model_name)

try:
    from agents.mcp_baseline_agent import MCPBaselineAgent
except ImportError:
    # Try direct import if agents module not found
    from mcp_baseline_agent import MCPBaselineAgent

async def run_simple_test():
    """Run a simple test of the MCP Baseline Agent."""
    print("=== Testing MCP Baseline Agent ===\n")


    
    try:
        # Initialize the agent
        agent = MCPBaselineAgent(
            server_path="excytin_bench_server.py",
            config_list=config_list,
            cache_seed=42,
            max_steps=8,
            submit_summary=False,
            temperature=0
        )
        
        print("Agent initialized successfully")
        
        # Run an episode
        print("Starting episode...")
        result = await agent.run_episode(
            attack="incident_5",
            q_idx=0,
            max_steps=8,
            split="test",
            use_full_db=False,
            layer="alert"
        )
        
        # Print results
        print("\n=== Episode Results ===")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Attack: {result['attack']}")
            print(f"Question Index: {result['question_idx']}")
            print(f"Total Steps: {result['total_steps']}")
            print(f"Question: {json.dumps(result['question'], indent=2)}")
            
            print("\n=== Episode History ===")
            for step in result['episode_history']:
                print(f"\n--- Step {step['step']} ---")
                
                # Show agent messages if available
                if 'messages' in step and step['messages']:
                    print("Agent Thinking:")
                    for msg in step['messages']:
                        if msg['role'] == 'assistant':
                            print(f"  {msg['content']}")
                
                print(f"Action: {step['action']}")
                print(f"Is Submit: {step['is_submit']}")
                if 'observation' in step:
                    obs_str = str(step['observation'])
                    if len(obs_str) > 300:
                        obs_str = obs_str[:300] + "..."
                    print(f"Observation: {obs_str}")
                print()
            
            if result['final_result']:
                print("=== Final Result ===")
                print(json.dumps(result['final_result'], indent=2, default=str))
            
            print("\n=== Usage Summary ===")
            print(json.dumps(result['usage_summary'], indent=2, default=str))
            
            # Show full conversation history for debugging
            print("\n=== Agent Conversation History ===")
            agent_logging = agent.get_logging()
            for i, msg in enumerate(agent_logging['messages']):
                print(f"\n--- Message {i+1} ({msg['role']}) ---")
                content = msg['content']
                if len(content) > 500:
                    content = content[:500] + "..."
                print(content)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def run_interactive_demo():
    """Run an interactive demo where you can step through manually."""
    print("=== Interactive MCP Baseline Agent Demo ===\n")
    
    try:
        agent = MCPBaselineAgent(
            server_path="excytin_bench_server.py",
            config_list=config_list,
            cache_seed=42,
            max_steps=15,
            submit_summary=True
        )
        
        print("Agent initialized successfully")
        
        # Get user input for episode parameters
        attack = input("Enter attack name (default: incident_5): ").strip() or "incident_5"
        q_idx_str = input("Enter question index (default: 0): ").strip() or "0"
        q_idx = int(q_idx_str)
        
        print(f"\nStarting interactive episode for {attack}, question {q_idx}")
        
        async with agent.mcp_client:
            # Initialize session
            if not await agent._initialize_session(attack, q_idx):
                print("Failed to initialize session")
                return
            
            # Get current question
            question = await agent._get_current_question()
            if question is None:
                print("Failed to get current question")
                return
            
            print(f"\nQuestion: {json.dumps(question, indent=2)}")
            
            # Extract question text
            if isinstance(question, dict):
                question_text = question.get('question', str(question))
            else:
                question_text = str(question)
            
            observation = f"Question: {question_text}"
            
            # Interactive loop
            step = 0
            while step < agent.max_steps:
                print(f"\n{'='*60}")
                print(f"STEP {step + 1}/{agent.max_steps}")
                print(f"{'='*60}")
                print(f"Current observation: {observation}")
                
                # Get action from agent (this will print the agent's thinking)
                action, is_submit = await agent.act(observation)
                
                print(f"\n[ACTION EXTRACTED]")
                print(f"Action: {action}")
                print(f"Is submit: {is_submit}")
                
                if is_submit:
                    # Submit answer
                    result = await agent._submit_answer(action)
                    print(f"\n[SUBMISSION RESULT]")
                    print(json.dumps(result, indent=2, default=str))
                    break
                else:
                    # Execute query
                    observation = await agent._execute_query(action)
                    print(f"\n[QUERY RESULT]")
                    print(f"Query: {action}")
                    print(f"Result: {observation}")
                
                step += 1
                
                # Ask user if they want to continue
                cont = input(f"\nPress Enter to continue, 'q' to quit: ").strip()
                if cont.lower() == 'q':
                    break
            
            print("\nDemo completed!")
        
    except Exception as e:
        print(f"Interactive demo failed with error: {e}")
        import traceback
        traceback.print_exc()

def print_usage():
    """Print usage information."""
    print("""
Usage: python test_mcp_baseline_agent.py [mode]

Modes:
  simple     - Run a simple automated test (default)
  interactive - Run an interactive demo
  help       - Show this help message

Before running, make sure to:
1. Update the EXAMPLE_CONFIG_LIST with your actual LLM configuration
2. Ensure the excytin_bench_server.py is in the same directory
3. Install required dependencies: fastmcp, autogen, etc.
""")

async def main():
    """Main function."""
    mode = "interactive"  # Default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "help":
        print_usage()
    elif mode == "interactive":
        await run_interactive_demo()
    elif mode == "simple":
        await run_simple_test()
    else:
        print(f"Unknown mode: {mode}")
        print_usage()

if __name__ == "__main__":
    asyncio.run(main())
