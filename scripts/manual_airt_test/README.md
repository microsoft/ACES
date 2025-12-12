# Manual AIRT Test Scripts

These scripts allow you to manually test the SABER AIRT (AI Red Team) demo by spinning up a server, creating episodes, and calling MCP tools.

## Prerequisites

- SABER server dependencies installed
- Docker running (for sandbox containers)
- `uv` package manager

## Quick Start

```bash
cd external/saber

# 1. Start the server and create episodes (runs in foreground)
uv run python scripts/manual_airt_test/start_server.py

# 2. In another terminal, call tools
uv run python scripts/manual_airt_test/call_tools.py demo
```

## Scripts

### `start_server.py`

Starts the SABER server and creates a session with blue and red team episodes.

```bash
uv run python scripts/manual_airt_test/start_server.py
```

This will:
1. Start the REST API server on port 8000
2. Start the MCP server on port 8001
3. Create a new session
4. Create a blue team episode with `bash` and `python` tools
5. Create a red team episode with `bash`, `python`, `inject_prompt`, and `get_target_transcript` tools
6. Save episode data to `episode_data.json`

**Keep this running** while using `call_tools.py`.

### `call_tools.py`

Calls MCP tools on the running server.

#### Commands

```bash
# Show help
uv run python scripts/manual_airt_test/call_tools.py --help

# List tools for a team
uv run python scripts/manual_airt_test/call_tools.py list blue
uv run python scripts/manual_airt_test/call_tools.py list red
uv run python scripts/manual_airt_test/call_tools.py list red --verbose

# Call a tool
uv run python scripts/manual_airt_test/call_tools.py call blue bash --command "echo hello"
uv run python scripts/manual_airt_test/call_tools.py call blue python --code "print('hello')"
uv run python scripts/manual_airt_test/call_tools.py call red inject_prompt --prompt "test injection"
uv run python scripts/manual_airt_test/call_tools.py call red get_target_transcript

# Show episode info
uv run python scripts/manual_airt_test/call_tools.py info

# Run demo (list all tools + demo calls)
uv run python scripts/manual_airt_test/call_tools.py demo

# Test inject flow (injects prompt and gets response)
uv run python scripts/manual_airt_test/call_tools.py test-inject --message "Ignore previous instructions"
uv run python scripts/manual_airt_test/call_tools.py test-inject --strategy restart --message "Fresh attack"  # Reset transcript first

# Interactive mode
uv run python scripts/manual_airt_test/call_tools.py interactive
```

#### Interactive Mode

In interactive mode, you can run commands without restarting:

```
> list blue
> list red
> call blue bash command="whoami"
> call red inject_prompt prompt="Ignore previous instructions"
> info
> quit
```

## Teams & Tools

### Blue Team (Defender)
- `bash` - Execute bash commands
- `python` - Execute Python code

### Red Team (Attacker)
- `bash` - Execute bash commands  
- `python` - Execute Python code
- `inject_prompt` - Inject adversarial prompts into target agent
- `get_target_transcript` - Get the target (blue) team's conversation transcript

## Blue Team Simulation

To test red team tools like `get_target_transcript`, the blue team needs to have a transcript with messages. Use these commands to simulate blue team activity:

```bash
# Push a message to blue team transcript (appends by default)
uv run python scripts/manual_airt_test/call_tools.py push blue --role user --content "Hello, I need help"
uv run python scripts/manual_airt_test/call_tools.py push blue --role assistant --content "I can help you with that."

# Push with restart strategy (resets to initial transcript first)
uv run python scripts/manual_airt_test/call_tools.py push blue --role user --content "New message" --strategy restart

# Get the blue team transcript (via WebSocket)
uv run python scripts/manual_airt_test/call_tools.py transcript blue

# Simulate a full agent turn (adds user + assistant messages)
uv run python scripts/manual_airt_test/call_tools.py simulate
```

### Transcript State

The transcript state determines what happens when red team tools execute:

- **WAITING_FOR_USER** - Last message is from assistant (no tool calls). Red team's `get_target_transcript` will return immediately.
- **WAITING_FOR_ASSISTANT** - Last message is from user. Red team's `get_target_transcript` will wait for assistant response.
- **WAITING_FOR_TOOLS** - Last message is from assistant with tool calls. Red team's `get_target_transcript` will wait for tool responses.

To put the blue team in WAITING_FOR_USER state, ensure the last message is an assistant message without tool calls.

## Files

- `start_server.py` - Server startup script
- `call_tools.py` - Tool calling script
- `episode_data.json` - Generated file with session/episode IDs (created by start_server.py)

## Troubleshooting

### "Episode data file not found"
Run `start_server.py` first to create the episodes.

### Red team only shows 2 tools
Make sure you're using the local `inspect-ai` package with the MCP caching fix. Check that `pyproject.toml` has:
```toml
[tool.uv.sources]
inspect-ai = { path = "../inspect_ai", editable = true }
```

### Connection refused
Make sure `start_server.py` is still running in another terminal.

### Docker errors
Ensure Docker is running and you have permissions to create containers.

### Red team `get_target_transcript` times out
The red team's `get_target_transcript` tool communicates with the server via a WebSocket daemon running inside the red sandbox container. Known issues:

1. **WebSocket connection drops**: The daemon's WebSocket connection to the server can become stale. Restart the container:
   ```bash
   docker restart ai-redteam-red-sandbox-<episode_id>
   ```

2. **Blue team not in WAITING_FOR_USER state**: The tool waits for the blue team to finish generating. Use the `push` command to add an assistant message (without tool calls) to put blue team in WAITING_FOR_USER state.

3. **Use `wait_for_user=false`**: Skip the wait by passing the argument:
   ```bash
   uv run python scripts/manual_airt_test/call_tools.py call red get_target_transcript -- --wait_for_user false
   ```
