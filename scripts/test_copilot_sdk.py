"""Simple test to verify Copilot SDK connection and messaging."""

import asyncio
import os
import subprocess
from pathlib import Path

# Load .env file
def load_dotenv():
    """Simple .env loader."""
    env_path = Path(__file__).parent.parent / ".env"
    print(f"Looking for .env at: {env_path}")
    if env_path.exists():
        print("Found .env file, loading...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()
        print("Loaded .env")
    else:
        print(".env file not found!")

load_dotenv()

def check_copilot_cli():
    """Check if Copilot CLI is available."""
    cli_paths = [
        os.path.expanduser("~/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli/copilot"),
        os.path.expanduser("~/.vscode/extensions/github.copilot-chat-*/copilotCli/copilot"),
        "/usr/local/bin/copilot",
    ]
    
    for path in cli_paths:
        from glob import glob
        matches = glob(path)
        if matches:
            cli = matches[0]
            if os.path.exists(cli):
                print(f"Found Copilot CLI at: {cli}")
                return cli
    
    print("WARNING: Copilot CLI not found in expected locations")
    return None

async def test_copilot_basic():
    """Test basic Copilot SDK functionality."""
    from copilot import CopilotClient
    
    print("=" * 60)
    print("Testing Copilot SDK Basic Connectivity")
    print("=" * 60)
    
    # Check Copilot CLI
    cli_path = check_copilot_cli()
    
    # Check environment
    print(f"\nAZUREAI_OPENAI_BASE_URL: {os.environ.get('AZUREAI_OPENAI_BASE_URL', 'NOT SET')[:50]}...")
    print(f"AZUREAI_OPENAI_API_KEY: {'SET' if os.environ.get('AZUREAI_OPENAI_API_KEY') else 'NOT SET'}")
    
    print("\n1. Creating CopilotClient...")
    client_options = {"auto_start": True}
    if cli_path:
        client_options["cli_path"] = cli_path
    client = CopilotClient(client_options)
    
    print("2. Starting client...")
    await client.start()
    print("   ✓ Client started")
    
    print("\n3. Creating session with BYOK provider...")
    
    # Build provider config from environment
    provider_config = {
        "type": "azure",
        "base_url": os.environ.get("AZUREAI_OPENAI_BASE_URL"),
        "api_key": os.environ.get("AZUREAI_OPENAI_API_KEY"),
        "azure": {
            "api_version": os.environ.get("AZUREAI_OPENAI_API_VERSION", "2025-03-01-preview")
        }
    }
    
    session_config = {
        "model": "gpt-4o",
        "tools": [],
        "system_message": {
            "mode": "append",
            "content": "You are a helpful assistant. Respond briefly.",
        },
        "provider": provider_config,
    }
    
    print(f"   Provider type: {provider_config['type']}")
    print(f"   Base URL: {provider_config['base_url'][:50]}..." if provider_config['base_url'] else "   Base URL: NOT SET")
    print(f"   Model: {session_config['model']}")
    
    try:
        session = await client.create_session(session_config)
        print("   ✓ Session created")
    except Exception as e:
        print(f"   ✗ Session creation failed: {e}")
        await client.stop()
        return
    
    print("\n4. Sending test message...")
    try:
        response = await asyncio.wait_for(
            session.send_and_wait({"prompt": "Say 'Hello World' and nothing else."}),
            timeout=30.0
        )
        print("   ✓ Response received")
        
        if response:
            print(f"\n   Response type: {type(response)}")
            print(f"   Has data: {hasattr(response, 'data')}")
            if hasattr(response, 'data'):
                print(f"   Data type: {type(response.data)}")
                print(f"   Has content: {hasattr(response.data, 'content')}")
                if hasattr(response.data, 'content'):
                    content = response.data.content
                    print(f"   Content type: {type(content)}")
                    print(f"   Content: {content}")
        else:
            print("   Response was None!")
            
    except asyncio.TimeoutError:
        print("   ✗ Request timed out after 30s")
    except Exception as e:
        print(f"   ✗ Error: {type(e).__name__}: {e}")
    
    print("\n5. Cleaning up...")
    try:
        await session.destroy()
        print("   ✓ Session destroyed")
    except Exception as e:
        print(f"   ✗ Session destroy error: {e}")
    
    await client.stop()
    print("   ✓ Client stopped")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_copilot_basic())
