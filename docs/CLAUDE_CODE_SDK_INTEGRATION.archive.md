# Claude Code SDK Integration - Progress Archive

**Date:** January 22, 2026
**SDK Version:** claude-code-sdk v0.0.25
**CLI Version:** @anthropic-ai/claude-code 2.1.15

## Overview

SABER integrates with Claude Code SDK to run an agentic loop that interacts with SABER's MCP tools in a sandbox environment. The goal is to have Claude Code use **only** SABER MCP tools (e.g., `mcp__saber__bash`, `mcp__saber__python`) rather than its built-in tools (e.g., `Bash`, `Write`, `Edit`).

## What Works

### MCP Connection
- HTTP transport to SABER MCP server at `http://localhost:8001/mcp` connects successfully
- Server reports capabilities: `{"hasTools":true,"hasPrompts":true,"hasResources":true}`
- Connection establishes in ~48ms with proper headers (session ID, episode ID, task ID)

### Debug Logging
- Enabled via `extra_args={"debug-to-stderr": None}` + `debug_stderr=<file_object>`
- Writes to `logs/claude_code/claude_code_debug_<timestamp>.log`
- Provides visibility into MCP initialization, tool hooks, and agent behavior

### Configuration
- `ClaudeCodeOptions` properly configured with:
  - `system_prompt` - Task instructions
  - `disallowed_tools` - Blocks built-in tools (Bash, Write, Edit, etc.)
  - `mcp_servers` - Points to SABER MCP endpoint
  - `permission_mode="bypassPermissions"` - SABER manages permissions

## What Doesn't Work

### Tool Restriction Bypass via Subagents
**Problem:** Claude Code agent uses the `Task` tool to spawn subagents that have access to built-in tools (including `Bash`), bypassing `disallowed_tools`.

**Evidence from debug logs:**
```
executePreToolHooks called for tool: Task
Getting matching hook commands for SubagentStart with query: Bash
```

The subagent is configured with `Bash` in its available tools, even though `Bash` is in the parent's `disallowed_tools`.

### `can_use_tool` Callback Not Being Invoked
**Problem:** The `can_use_tool` callback was defined but never called.

**Root Cause:** The SDK requires **streaming mode** (AsyncIterable prompt) for `can_use_tool` to work. From SDK source:
```python
if self.options.can_use_tool:
    # canUseTool callback requires streaming mode (AsyncIterable prompt)
    if isinstance(prompt, str):
        raise ValueError(
            "can_use_tool callback requires streaming mode. "
            "Please provide prompt as an AsyncIterable instead of a string."
        )
```

**Original code pattern (broken):**
```python
async with ClaudeSDKClient(options=options) as client:
    await client.query(initial_prompt)  # String prompt!
```

The context manager calls `connect()` with `None`, which bypasses the validation. Then `query()` is called with a string.

**Fix implemented:**
```python
async def initial_prompt_stream():
    yield {
        "type": "user",
        "message": {"role": "user", "content": initial_prompt},
        "parent_tool_use_id": None,
        "session_id": "saber-session",
    }

client = ClaudeSDKClient(options=options)
await client.connect(initial_prompt_stream())  # AsyncIterable!
```

## Challenges

### 1. SDK Documentation Gaps
- No clear documentation on how `can_use_tool` interacts with subagents
- Had to reverse-engineer SDK source to understand streaming mode requirement
- Unclear if `can_use_tool` applies to subagent tool use or only main agent

### 2. Subagent Tool Inheritance
- Subagents spawned by `Task` tool appear to have their own tool configuration
- Debug logs show `SubagentStart with query: Bash` suggesting subagent gets Bash access
- The `disallowed_tools` from parent may not propagate to subagents

### 3. MCP Tool Discovery
- When `allowed_tools` was set (even to empty list), MCP tools weren't accessible
- Removing `allowed_tools` entirely lets Claude Code auto-discover from `mcp_servers`
- Tool naming: MCP server exposes `bash`, Claude Code sees it as `mcp__saber__bash`

### 4. Permission Mode Interaction
- `permission_mode="bypassPermissions"` may interfere with `can_use_tool` callback
- Need to verify if both can work together

## Current State of Code

**File:** `external/saber/src/saber/inspect_ai/agents/registry/claude_code.py`

### `disallowed_tools` List
```python
disallowed_tools = [
    # File system tools
    "Bash", "Write", "Edit", "MultiEdit", "Read", "Glob", "Grep", "LS",
    # Web tools
    "WebFetch", "WebSearch",
    # Note/todo tools
    "TodoRead", "TodoWrite", "NotebookRead", "NotebookEdit",
    # MCP introspection tools
    "ListMcpResources", "ReadMcpResource", "Skill",
    # Plan approval
    "Plan",
    # Note: Task is NOT disabled - we allow subagents
]
```

### `can_use_tool` Callback
```python
async def enforce_tool_restrictions(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    # Allow SABER MCP tools (mcp__saber__*)
    if tool_name.startswith("mcp__saber__"):
        logger.debug(f"Tool '{tool_name}' allowed (mcp__saber__ prefix)")
        return PermissionResultAllow()

    # Block explicitly disallowed tools
    if tool_name in disallowed_tools_set:
        return PermissionResultDeny(
            message=f"Tool '{tool_name}' is not available. Use SABER MCP tools instead.",
            interrupt=False,
        )

    # Block all other non-MCP tools
    return PermissionResultDeny(
        message=f"Tool '{tool_name}' is not available in SABER. Use SABER MCP tools.",
        interrupt=False,
    )
```

### ClaudeCodeOptions
```python
options = ClaudeCodeOptions(
    system_prompt=system_prompt,
    disallowed_tools=disallowed_tools,
    mcp_servers={"saber": mcp_server_config},
    # NOTE: No allowed_tools - let Claude Code auto-discover MCP tools
    permission_mode="bypassPermissions",
    extra_args={"debug-to-stderr": None},
    debug_stderr=debug_log_file,
    can_use_tool=enforce_tool_restrictions,
)
```

## Next Steps

### Immediate (Verify Fix)
1. Run test with streaming mode fix: `uv run inspect eval domains/excytin_demo -T agent=claude_code --limit 1`
2. Check debug log for `can_use_tool` callback invocations
3. Verify subagent tool use is being intercepted

### If `can_use_tool` Still Not Working for Subagents
1. **Option A:** Add `Task` to `disallowed_tools` (prevents subagents entirely)
2. **Option B:** Investigate SDK hooks system (`SubagentStart` event) as alternative
3. **Option C:** File issue with Anthropic about subagent tool inheritance

### Future Improvements
1. Add system prompt guidance directing agent to use `mcp__saber__*` tools
2. Implement retry logic when tool is denied (help agent recover)
3. Consider using `PermissionResultDeny(interrupt=True)` to force agent to stop on violation
4. Test with `permission_mode="default"` to see if it enables callback without bypassing

## Reference: Working Test Script Pattern

From `scripts/test_cc_mcp.py` (known working):
```python
options = ClaudeCodeOptions(
    disallowed_tools=[],  # Empty - let can_use_tool handle it
    mcp_servers={"saber": mcp_config},
    permission_mode="bypassPermissions",
)

async for message in query(prompt="Use the bash tool...", options=options):
    print(message)
```

Key difference: Uses `query()` function (one-shot) rather than `ClaudeSDKClient` (interactive). The `query()` function may handle streaming mode internally.

## Debug Log Locations
- Claude Code debug: `external/saber/logs/claude_code/claude_code_debug_*.log`
- SABER logs: Check Inspect AI log viewer

## Useful Commands
```bash
# Run test
cd /home/kyledeprow/repos/oss_saber_dev
uv run inspect eval domains/excytin_demo -T agent=claude_code --limit 1

# Check SDK version
uv run python -c "import claude_code_sdk; print(claude_code_sdk.__version__)"

# Verify module loads
uv run python -c "from saber.inspect_ai.agents.registry.claude_code import _run_claude_code_agent; print('OK')"
```
