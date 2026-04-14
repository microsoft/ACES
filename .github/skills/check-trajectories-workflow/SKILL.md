---
name: check-trajectories-workflow
description: Analyze agent trajectories from evaluation log files to detect external failures, formatting issues, reward hacking, and ethical refusals. Use when user asks to check/analyze agent trajectories. Trigger when the user asks you to run the "Check Agent Trajectories" workflow.
---

# Check Agent Trajectories Workflow

Analyze agent behavior in evaluation logs to detect anomalies and failure patterns.

## Step 1: Get Log File

Ask the user for the `.eval` log file path, or find the most recent one:

```bash
ls -lt logs/*.eval | head -5
```

## Step 2: Setup Output Directory

```bash
mkdir -p agent_artifacts/trajectory_analysis/<domain>/
```

## Step 3: Extract and Analyze Trajectories

Parse each sample's message history to detect patterns:

```python
import json, zipfile
from collections import Counter

log_file = "<path_to_eval_file>"
findings = {
    "external_failures": [],
    "formatting_failures": [],
    "reward_hacking": [],
    "ethical_refusals": [],
    "tool_errors": [],
    "stuck_loops": [],
}

with zipfile.ZipFile(log_file, "r") as zf:
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))
        task_id = sample.get("id", "unknown")
        messages = sample.get("messages", [])

        tool_calls = []

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "assistant" and any(phrase in content.lower() for phrase in [
                "i cannot", "i can't", "i'm unable", "i refuse",
                "ethical concerns", "not appropriate"
            ]):
                findings["ethical_refusals"].append({
                    "task": task_id, "content": content[:200]
                })

            for tc in msg.get("tool_calls", []):
                tool_calls.append(tc.get("function", "unknown"))
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    findings["formatting_failures"].append({
                        "task": task_id, "tool": tc.get("function"),
                        "issue": "Arguments are string, not dict"
                    })

            if role == "tool":
                if any(err in content.lower() for err in [
                    "error", "traceback", "exception", "permission denied",
                    "no such file", "connection refused"
                ]):
                    findings["tool_errors"].append({
                        "task": task_id, "content": content[:200]
                    })

        if len(tool_calls) >= 3:
            for i in range(len(tool_calls) - 2):
                if tool_calls[i] == tool_calls[i+1] == tool_calls[i+2]:
                    findings["stuck_loops"].append({
                        "task": task_id, "tool": tool_calls[i],
                        "count": sum(1 for t in tool_calls if t == tool_calls[i])
                    })
                    break
```

## Step 4: Classify Findings

For each finding category, assess severity:

| Category | Low | Medium | High |
|----------|-----|--------|------|
| External Failures | Transient network hiccup | Repeated Docker timeout | Sandbox completely broken |
| Formatting Failures | Single malformed call | >10% of calls malformed | Agent can't use tools at all |
| Reward Hacking | Minor shortcut | Scores inflated by gaming | Completely bypasses evaluation |
| Ethical Refusals | Overly cautious phrasing | Refuses key task steps | Refuses entire task |
| Tool Errors | Recoverable error | Blocks progress | Cascading failures |
| Stuck Loops | Brief retry loop | Moderate repetition | Burns entire token budget |

## Step 5: Write Analysis

Create `agent_artifacts/trajectory_analysis/<domain>/<domain>_<model>_ANALYSIS.md`:

```markdown
# Trajectory Analysis: <domain> / <model>

## Summary
- Log file: `<path>`
- Total samples: N
- Samples with issues: N (XX%)

## Error Rate
- External failures: N (XX%)
- Formatting failures: N (XX%)
- Reward hacking attempts: N (XX%)
- Ethical refusals: N (XX%)
- Tool errors: N (XX%)
- Stuck loops: N (XX%)

## Overall Error Rate: XX%
Target: ≤10%, Ideal: ≤5%

## Detailed Findings

### External Failures
| Task | Issue | Severity |
|------|-------|----------|
| ... | ... | Low/Med/High |

### Recommendations
1. ...
```

## Step 6: Report

Highlight the overall error rate and any High severity findings to the user.
