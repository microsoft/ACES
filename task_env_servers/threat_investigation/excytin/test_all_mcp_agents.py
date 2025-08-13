#!/usr/bin/env python3
"""
Test script for all MCP-based security analysis agents.
This script demonstrates how to use the various agents to interact with the Excytin Bench MCP server.
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

# Import all MCP agents
from agents.mcp_baseline_agent import MCPBaselineAgent
from agents.mcp_react_agent import MCPReActAgent
from agents.mcp_cheating_agent import MCPCheatingAgent
from agents.mcp_react_reflexion_agent import MCPReActReflexionAgent
from agents.mcp_multi_model_baseline_agent import MCPMultiModelBaselineAgent

# Configuration
model_name = "gpt-4o"  # Default eval model, can be overridden
config_list = filter_config_list(CONFIG_LIST, model_name)

# For multi-model agent
model_name_master = "gpt-4o"  # Strong model
model_name_slave = "o4-mini"  # Fast/cheap model
config_list_master = filter_config_list(CONFIG_LIST, model_name_master)
config_list_slave = filter_config_list(CONFIG_LIST, model_name_slave)

async def test_agent(agent_class, agent_name, init_args=None, **kwargs):
    """Test a specific agent."""
    print(f"\n{'='*80}")
    print(f"TESTING {agent_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Initialize the agent
        if init_args:
            agent = agent_class(**init_args)
        else:
            agent = agent_class(
                server_path="excytin_bench_server.py",
                config_list=config_list,
                cache_seed=42,
                max_steps=5,  # Keep short for testing
                submit_summary=False,
                temperature=0
            )
        
        print(f"{agent_name} initialized successfully")
        
        # Run an episode
        print(f"Starting {agent_name} episode...")
        result = await agent.run_episode(
            attack="incident_5",
            q_idx=0,
            max_steps=5,
            split="test",
            use_full_db=False,
            layer="alert",
            **kwargs
        )
        
        # Print results
        print(f"\n=== {agent_name} Episode Results ===")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Attack: {result['attack']}")
            print(f"Question Index: {result['question_idx']}")
            print(f"Total Steps: {result['total_steps']}")
            print(f"Question: {json.dumps(result['question'], indent=2)}")
            
            print(f"\n=== {agent_name} Episode History ===")
            for step in result['episode_history']:
                print(f"\n--- Step {step['step']} ---")
                
                # Show agent messages if available
                if 'messages' in step and step['messages']:
                    print("Agent Thinking:")
                    for msg in step['messages']:
                        if msg['role'] == 'assistant':
                            content = msg['content']
                            if len(content) > 200:
                                content = content[:200] + "..."
                            print(f"  {content}")
                
                print(f"Action: {step['action']}")
                print(f"Is Submit: {step['is_submit']}")
                
                # Show model info for multi-model agent
                if 'model_used' in step:
                    print(f"Model Used: {step['model_used']} ({step['model_name']})")
                
                if 'observation' in step:
                    obs_str = str(step['observation'])
                    if len(obs_str) > 200:
                        obs_str = obs_str[:200] + "..."
                    print(f"Observation: {obs_str}")
                print()
            
            if result.get('final_result'):
                print(f"=== {agent_name} Final Result ===")
                print(json.dumps(result['final_result'], indent=2, default=str))
            
            print(f"\n=== {agent_name} Usage Summary ===")
            print(json.dumps(result['usage_summary'], indent=2, default=str))
            
            # Show additional info for specific agents
            if 'incident_info' in result:
                print(f"\n=== Incident Info (Cheating Agent) ===")
                print(json.dumps(result['incident_info'], indent=2, default=str))
            
            if 'replay_buffer_size' in result:
                print(f"\n=== Replay Buffer Size (Reflexion Agent) ===")
                print(f"Buffer size: {result['replay_buffer_size']}")
            
            if 'master_usage' in result and 'slave_usage' in result:
                print(f"\n=== Multi-Model Usage Breakdown ===")
                print(f"Master Usage: {json.dumps(result['master_usage'], indent=2, default=str)}")
                print(f"Slave Usage: {json.dumps(result['slave_usage'], indent=2, default=str)}")
                print(f"Switch Interval: {result['switch_interval']}")
        
        return True
        
    except Exception as e:
        print(f"{agent_name} test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run tests for all MCP agents."""
    print("=== Testing All MCP Security Analysis Agents ===\n")
    
    results = {}
    
    # Test Baseline Agent
    results["baseline"] = await test_agent(
        MCPBaselineAgent, 
        "MCP Baseline Agent"
    )
    
    # Test ReAct Agent
    results["react"] = await test_agent(
        MCPReActAgent, 
        "MCP ReAct Agent"
    )
    
    # Test Cheating Agent
    results["cheating"] = await test_agent(
        MCPCheatingAgent, 
        "MCP Cheating Agent"
    )
    
    # Test ReAct Reflexion Agent
    results["reflexion"] = await test_agent(
        MCPReActReflexionAgent, 
        "MCP ReAct Reflexion Agent"
    )
    
    # Test Multi-Model Baseline Agent
    multi_model_args = {
        "server_path": "excytin_bench_server.py",
        "config_list_master": config_list_master,
        "config_list_slave": config_list_slave,
        "cache_seed": 42,
        "max_steps": 5,
        "submit_summary": False,
        "temperature": 0,
        "switch_interval": 2
    }
    results["multi_model"] = await test_agent(
        MCPMultiModelBaselineAgent, 
        "MCP Multi-Model Baseline Agent",
        init_args=multi_model_args
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    for agent_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{agent_name.upper()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

async def run_single_test(agent_type: str):
    """Run test for a single agent type."""
    agent_map = {
        "baseline": (MCPBaselineAgent, "MCP Baseline Agent"),
        "react": (MCPReActAgent, "MCP ReAct Agent"),
        "cheating": (MCPCheatingAgent, "MCP Cheating Agent"),
        "reflexion": (MCPReActReflexionAgent, "MCP ReAct Reflexion Agent"),
        "multi_model": (MCPMultiModelBaselineAgent, "MCP Multi-Model Baseline Agent")
    }
    
    if agent_type not in agent_map:
        print(f"Unknown agent type: {agent_type}")
        print(f"Available types: {', '.join(agent_map.keys())}")
        return
    
    agent_class, agent_name = agent_map[agent_type]
    
    if agent_type == "multi_model":
        multi_model_args = {
            "server_path": "excytin_bench_server.py",
            "config_list_master": config_list_master,
            "config_list_slave": config_list_slave,
            "cache_seed": 42,
            "max_steps": 8,
            "submit_summary": False,
            "temperature": 0,
            "switch_interval": 3
        }
        await test_agent(agent_class, agent_name, init_args=multi_model_args)
    else:
        await test_agent(agent_class, agent_name)

async def run_interactive_demo(agent_type: str):
    """Run an interactive demo for a specific agent."""
    agent_map = {
        "baseline": (MCPBaselineAgent, "MCP Baseline Agent"),
        "react": (MCPReActAgent, "MCP ReAct Agent"),
        "cheating": (MCPCheatingAgent, "MCP Cheating Agent"),
        "reflexion": (MCPReActReflexionAgent, "MCP ReAct Reflexion Agent"),
        "multi_model": (MCPMultiModelBaselineAgent, "MCP Multi-Model Baseline Agent")
    }
    
    if agent_type not in agent_map:
        print(f"Unknown agent type: {agent_type}")
        print(f"Available types: {', '.join(agent_map.keys())}")
        return
    
    agent_class, agent_name = agent_map[agent_type]
    
    print(f"=== Interactive {agent_name} Demo ===\n")
    
    try:
        # Initialize agent based on type
        if agent_type == "multi_model":
            agent = agent_class(
                server_path="excytin_bench_server.py",
                config_list_master=config_list_master,
                config_list_slave=config_list_slave,
                cache_seed=42,
                max_steps=15,
                submit_summary=True,
                switch_interval=3
            )
        else:
            agent = agent_class(
                server_path="excytin_bench_server.py",
                config_list=config_list,
                cache_seed=42,
                max_steps=15,
                submit_summary=True
            )
        
        print(f"{agent_name} initialized successfully")
        
        # Get user input for episode parameters
        attack = input("Enter attack name (default: incident_5): ").strip() or "incident_5"
        q_idx_str = input("Enter question index (default: 0): ").strip() or "0"
        q_idx = int(q_idx_str)
        
        print(f"\nStarting interactive episode for {attack}, question {q_idx}")
        
        # Run the episode - the agent's run_episode method will handle the interactive display
        result = await agent.run_episode(
            attack=attack,
            q_idx=q_idx,
            max_steps=15,
            split="test",
            use_full_db=False,
            layer="alert"
        )
        
        print(f"\n=== {agent_name} Demo Completed ===")
        print(f"Total steps: {result.get('total_steps', 'N/A')}")
        if result.get('final_result'):
            print(f"Final result: {json.dumps(result['final_result'], indent=2, default=str)}")
        
    except Exception as e:
        print(f"Interactive demo failed with error: {e}")
        import traceback
        traceback.print_exc()

def print_usage():
    """Print usage information."""
    print("""
Usage: python test_all_mcp_agents.py [mode] [agent_type]

Modes:
  all        - Run tests for all agents (default)
  single     - Run test for a single agent type
  interactive - Run an interactive demo for a single agent type
  help       - Show this help message

Agent Types (for single/interactive modes):
  baseline   - MCP Baseline Agent
  react      - MCP ReAct Agent (with examples)
  cheating   - MCP Cheating Agent (with incident hints)
  reflexion  - MCP ReAct Reflexion Agent (with self-reflection)
  multi_model - MCP Multi-Model Baseline Agent (switches between models)

Examples:
  python test_all_mcp_agents.py all
  python test_all_mcp_agents.py single react
  python test_all_mcp_agents.py interactive baseline

Before running, make sure to:
1. Update the CONFIG_LIST with your actual LLM configuration
2. Ensure the excytin_bench_server.py is in the same directory
3. Install required dependencies: fastmcp, autogen, etc.
""")

async def main():
    """Main function."""
    mode = "all"  # Default mode
    agent_type = None
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if len(sys.argv) > 2:
        agent_type = sys.argv[2].lower()
    
    if mode == "help":
        print_usage()
    elif mode == "all":
        await run_all_tests()
    elif mode == "single":
        if not agent_type:
            print("Error: Agent type required for single mode")
            print("Available types: baseline, react, cheating, reflexion, multi_model")
            return
        await run_single_test(agent_type)
    elif mode == "interactive":
        if not agent_type:
            print("Error: Agent type required for interactive mode")
            print("Available types: baseline, react, cheating, reflexion, multi_model")
            return
        await run_interactive_demo(agent_type)
    else:
        print(f"Unknown mode: {mode}")
        print_usage()

if __name__ == "__main__":
    asyncio.run(main())
