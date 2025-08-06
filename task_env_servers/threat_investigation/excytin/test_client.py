import asyncio
import json
from fastmcp import Client

# Initialize client pointing to the excytin bench server
client = Client("excytin_bench_server.py")

def clean_table_name(table):
    """Clean table name by removing various characters and quotes."""
    return str(table).strip("(),").replace("'", "").replace('"', '').replace('[', '').replace(']', '').strip()

async def test_all_tools():
    """Test all available tools in the Excytin Bench MCP server."""
    async with client:
        print("=== Testing Excytin Bench MCP Server ===\n")
        
        # 1. Initialize context
        print("1. Initializing context...")
        try:
            result = await client.call_tool("initialize_context", {
                "attack": "incident_5",
                "max_steps": 15,
                "split": "test",
                "use_full_db": False,
                "layer": "alert",
                "q_idx": 0
            })
            # Extract the actual data from CallToolResult
            result_data = result.data if hasattr(result, 'data') else result
            print(f"✓ Context initialized: {result_data}")
        except Exception as e:
            print(f"✗ Failed to initialize context: {e}")
            return
        
        print("\n" + "="*50 + "\n")
        
        # 2. Get current question
        print("2. Getting current question...")
        try:
            result = await client.call_tool("get_current_question", {})
            # Extract the actual data from CallToolResult
            if hasattr(result, 'data'):
                result_data = result.data
            elif hasattr(result, 'structured_content'):
                result_data = result.structured_content
            else:
                result_data = result
            
            print(f"✓ Current question: {json.dumps(result_data, indent=2)}")
            current_question = result_data.get('question', {})
        except Exception as e:
            print(f"✗ Failed to get current question: {e}")
            return
        
        print("\n" + "="*50 + "\n")
        
        # 3. Get database schema (all tables)
        print("3. Getting database schema...")
        try:
            result = await client.call_tool("get_database_schema", {})
            result_data = result.data if hasattr(result, 'data') else result
            print(f"✓ Database tables: {result_data}")
            tables = result_data.get('table_names', [])
        except Exception as e:
            print(f"✗ Failed to get database schema: {e}")
            tables = []
        
        print("\n" + "="*50 + "\n")
        
        # 4. Test SQL queries
        print("4. Testing SQL queries...")
        
        # Basic query to show tables
        try:
            result = await client.call_tool("query_sql_database", {
                "query": "SHOW TABLES;"
            })
            result_data = result.data if hasattr(result, 'data') else result
            print(f"✓ SHOW TABLES query: {json.dumps(result_data, indent=2)}")
        except Exception as e:
            print(f"✗ Failed SHOW TABLES query: {e}")
        
        # Query a specific table if available
        if tables and len(tables) > 0:
            table_name = clean_table_name(tables[0])
            try:
                result = await client.call_tool("query_sql_database", {
                    "query": f"SELECT * FROM {table_name} LIMIT 3;"
                })
                result_data = result.data if hasattr(result, 'data') else result
                print(f"✓ Sample data query: {json.dumps(result_data, indent=2)}")
            except Exception as e:
                print(f"✗ Failed sample data query: {e}")
        
        print("\n" + "="*50 + "\n")
        
        # 5. Test context resource
        print("5. Testing context resource...")
        try:
            result = await client.read_resource("session://context")
            result_data = result.data if hasattr(result, 'data') else result
            print(f"✓ Context resource: {result_data}")
        except Exception as e:
            print(f"✗ Failed to get context resource: {e}")
        
        print("\n" + "="*50 + "\n")
        
        return current_question

async def run_simple_episode():
    """Run a simple episode trying to answer question with q_idx=0."""
    async with client:
        print("=== Running Simple Episode (Question 0) ===\n")
        
        # Initialize context
        print("Initializing context for question 0...")
        try:
            result = await client.call_tool("initialize_context", {
                "attack": "incident_5",
                "max_steps": 10,
                "split": "test",
                "use_full_db": False,
                "layer": "alert",
                "q_idx": 0
            })
            result_data = result.data if hasattr(result, 'data') else result
            print(f"Context: {result_data}")
        except Exception as e:
            print(f"Failed to initialize context: {e}")
            return
        
        # Get the current question
        print("\nGetting question...")
        try:
            result = await client.call_tool("get_current_question", {})
            result_data = result.data if hasattr(result, 'data') else result
            question = result_data.get('question', {})
            print(f"Question: {json.dumps(question, indent=2)}")
            
            # Extract question text
            question_text = question.get('question', '') if isinstance(question, dict) else str(question)
            print(f"\nQuestion text: {question_text}")
        except Exception as e:
            print(f"Failed to get question: {e}")
            return
        
        # Explore the database
        print("\nExploring database...")
        
        # Get all tables
        try:
            result = await client.call_tool("get_database_schema", {})
            result_data = result.data if hasattr(result, 'data') else result
            tables = result_data.get('table_names', [])
            print(f"Available tables: {tables}")
        except Exception as e:
            print(f"Failed to get tables: {e}")
            return
        
        # Look for security-related tables
        security_tables = []
        for table in tables:
            table_str = clean_table_name(table)
            if any(keyword in table_str.lower() for keyword in ['security', 'alert', 'incident', 'threat']):
                security_tables.append(table_str)
        
        print(f"Security-related tables: {security_tables}")
        
        # Query some relevant tables
        for table in security_tables[:3]:  # Limit to first 3 tables
            try:
                print(f"\nQuerying {table}...")
                result = await client.call_tool("query_sql_database", {
                    "query": f"SELECT * FROM {table} LIMIT 5;"
                })
                result_data = result.data if hasattr(result, 'data') else result
                observation = result_data.get('observation', 'No data')
                print(f"Sample data from {table}: {str(observation)[:200]}...")
            except Exception as e:
                print(f"Failed to query {table}: {e}")
        
        # Submit a simple answer (this is just a test)
        print("\nSubmitting a test answer...")
        try:
            result = await client.call_tool("submit_answer", {
                "answer": "Based on initial investigation, this appears to be a security incident requiring further analysis."
            })
            result_data = result.data if hasattr(result, 'data') else result
            print(f"Answer submission result: {json.dumps(result_data, indent=2)}")
        except Exception as e:
            print(f"Failed to submit answer: {e}")
        
        # Check final context
        print("\nChecking final context...")
        try:
            result = await client.read_resource("session://context")
            result_data = result.data if hasattr(result, 'data') else result
            print(f"Final context: {result_data}")
        except Exception as e:
            print(f"Failed to get final context: {e}")

async def main():
    """Main function to run all tests."""
    print("Starting Excytin Bench MCP Server Tests...\n")
    
    # Test all tools
    current_question = await test_all_tools()
    
    print("\n" + "="*70 + "\n")
    
    # Run a simple episode
    await run_simple_episode()
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())