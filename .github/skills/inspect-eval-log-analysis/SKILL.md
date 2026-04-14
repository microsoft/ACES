---
name: inspect-eval-log-analysis
description: Guide for parsing and analyzing inspect_ai .eval log files. Use this when asked to interpret eval results, extract tool calls, find scores, or investigate agent behavior from .eval logs.
---

To parse and analyze inspect_ai `.eval` log files, follow this process:

## 1. Understanding the `.eval` File Format

`.eval` files are **ZIP archives** (not plain text). They contain structured JSON:

```text
<hash>.eval (ZIP)
├── header.json                    # Eval metadata
├── _journal/
│   ├── start.json                 # Eval config, plan, model info
│   └── summaries/
│       └── 1.json                 # Per-epoch summary
├── samples/
│   └── <task_id>_epoch_1.json     # Full sample data per task
├── summaries.json                 # Aggregated score summaries
└── reductions.json                # Score reduction results
```

**Never try to read `.eval` files as plain text.** Always use `zipfile`:

```python
import json, zipfile

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    print(zf.namelist())  # See all files inside
```

## 2. Extract Sample Data (Messages, Tool Calls, Scores)

The richest data is in the `samples/` JSON files:

```python
import json, zipfile

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))

        # Key fields:
        # sample["id"]         — task ID
        # sample["messages"]   — full message history
        # sample["scores"]     — dict of scorer_name → score result
        # sample["metadata"]   — task metadata
```

## 3. Analyze Tool Calls from Messages

```python
import json, zipfile

def extract_tool_calls(eval_path):
    """Extract all tool calls from an eval log."""
    results = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/"):
                continue
            sample = json.loads(zf.read(name))
            for msg in sample.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                for tc in msg.get("tool_calls", []):
                    if not isinstance(tc, dict):
                        continue
                    results.append({
                        "id": tc.get("id"),
                        "function": tc.get("function"),
                        "arguments": tc.get("arguments", {}),
                        "type": tc.get("type"),
                    })
    return results

calls = extract_tool_calls("logs/<file>.eval")
for c in calls:
    fn = c["function"]
    args = c["arguments"]
    if isinstance(args, dict):
        cmd = args.get("cmd", args.get("code", ""))
    print(f"{fn}: {cmd[:80]}")
```

## 4. Analyze Scores

```python
import json, zipfile

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))
        task_id = sample.get("id", "unknown")
        for scorer_name, score_data in sample.get("scores", {}).items():
            value = score_data.get("value")
            explanation = score_data.get("explanation", "")[:200]
            print(f"{task_id} | {scorer_name}: {value} — {explanation}")
```

## 5. Quick Summary

```python
import json, zipfile
from collections import Counter

with zipfile.ZipFile("logs/<file>.eval", "r") as zf:
    scores = Counter()
    total = 0
    for name in zf.namelist():
        if not name.startswith("samples/"):
            continue
        sample = json.loads(zf.read(name))
        total += 1
        for scorer_name, score_data in sample.get("scores", {}).items():
            v = score_data.get("value")
            if v is not None:
                scores[scorer_name] += float(v) if isinstance(v, (int, float)) else 0

    print(f"Total samples: {total}")
    for scorer, total_score in scores.items():
        print(f"  {scorer}: avg={total_score/total:.3f}")
```
