---
name: inspect-eval-debugging
description: Guide for debugging inspect_ai evaluation failures, score issues, and model behavior. Use this when eval results are unexpected, scores are wrong, scoring fails, or model output appears corrupted.
---

To debug inspect_ai evaluation issues in SABER, follow this systematic process:

## 1. Triage: Identify the Failure Category

Check the eval summary output. Common categories:

| Symptom | Likely Cause | Section |
|---------|-------------|---------|
| Score is `nan` for all scorers | Agent never reached scoring (sandbox error, tool failure) | §2 |
| Score is `0.0` but expected higher | Scorer ran but agent's work was incorrect or scorer has a bug | §3 |
| Score differs across runs of same model | Non-deterministic model behavior | §4 |
| Eval crashes with traceback | Code error in scoring/tools or inspect_ai | §5 |
| Docker/sandbox errors | Container issues | §6 |

## 2. Debug Agent Behavior (Score is `nan` or Unexpected)

Parse the eval log to see exactly what the agent did:

```python
import json, zipfile

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))

        print(f"Task: {sample.get('id')}")
        print(f"Scores: {sample.get('scores', {}).keys()}")

        for i, msg in enumerate(sample.get("messages", [])):
            role = msg.get("role", "?")
            if role == "assistant":
                tcs = msg.get("tool_calls", [])
                if tcs:
                    for tc in tcs:
                        fn = tc.get("function", "?")
                        args = tc.get("arguments", {})
                        cmd = args.get("cmd", "") if isinstance(args, dict) else str(args)[:100]
                        print(f"  [{i}] TOOL: {fn}({cmd[:80]})")
                else:
                    content = msg.get("content", "")
                    text = content if isinstance(content, str) else str(content)[:100]
                    print(f"  [{i}] ASSISTANT: {text[:100]}")
            elif role == "tool":
                text = msg.get("content", "")[:80]
                print(f"  [{i}] TOOL_RESULT: {text}")
```

## 3. Debug Scorer Issues (Score is 0.0 Unexpectedly)

### 3a. Check the score explanation

```python
import json, zipfile

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))
        for scorer_name, score_data in sample.get("scores", {}).items():
            print(f"--- {scorer_name} ---")
            print(f"  value: {score_data.get('value')}")
            print(f"  explanation: {score_data.get('explanation', 'none')[:500]}")
```

### 3b. Test the scorer in isolation

Write a standalone test that calls the scorer directly against a known-good input. See `tests/scoring/` for examples.

### 3c. Add debug logging

Scoring code lives in `src/saber/scoring/` or `domains/<domain>/scoring/`. Run with `INSPECT_LOG_LEVEL=info`:

```bash
INSPECT_LOG_LEVEL=info uv run inspect eval domains/<domain> --model <model> --limit 1
```

## 4. Debug Model Behavior

Some models produce malformed tool call arguments. Check for:
- Trailing `}` characters in bash commands
- JSON parse errors in tool arguments
- Repeated identical tool calls (stuck loops)

## 5. Debug Code Errors

If eval crashes with a traceback:
1. Read the full traceback
2. Identify whether it's in `src/saber/`, `domains/`, or `inspect_ai`
3. Add a targeted test to reproduce
4. Fix and re-run with `--limit 1`

## 6. Debug Docker/Sandbox Issues

```bash
# Check running containers
docker ps -a --filter "name=saber"

# Check container logs
docker logs <container_id>

# Remove stale containers
docker ps -a --filter "name=saber" -q | xargs -r docker rm -f

# Rebuild images
uv run inspect eval domains/<domain> --model <model> -T rebuild_all=true --limit 1
```
