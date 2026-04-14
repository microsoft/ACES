---
name: eval-report-workflow
description: Create an evaluation report by selecting models, running evaluations, and formatting results tables. Use when user asks to make/create/generate an evaluation report. Trigger when the user asks you to run the "Evaluation Report" workflow.
---

# Evaluation Report Workflow

Create an evaluation report for a domain by running evals across models and formatting results.

## Step 1: Select Models

Ask the user which models to evaluate, or recommend from available options:

**Common models:**
| Model | Provider | Notes |
|-------|----------|-------|
| `openai/azure/gpt-4.1` | Azure OpenAI | Good baseline |
| `openai/azure/gpt-4.1-mini` | Azure OpenAI | Faster, cheaper |
| `openai/gpt-4` | OpenAI | Standard |

Recommend at least 2 models for comparison.

## Step 2: Initial Test Run

Run a quick test with 1 sample to estimate time and verify setup:

```bash
uv run inspect eval domains/<domain> --model <model> --limit 1 -T task_filter="<simple_task>"
```

Verify the eval completes successfully before running the full suite.

## Step 3: Run Full Evaluations

For each model:

```bash
uv run inspect eval domains/<domain> --model <model> 2>&1 | tee /tmp/eval_<model_short>.log
```

Note the log file paths from the output.

## Step 4: Parse Results

Extract results from the `.eval` log files:

```python
import json, zipfile, os

log_dir = "logs"
results = []

for f in sorted(os.listdir(log_dir)):
    if not f.endswith(".eval") or "<domain>" not in f:
        continue
    with zipfile.ZipFile(f"{log_dir}/{f}", "r") as zf:
        header = json.loads(zf.read("header.json"))
        model = header.get("eval", {}).get("model", "unknown")

        scores = {}
        for name in zf.namelist():
            if not name.startswith("samples/"):
                continue
            sample = json.loads(zf.read(name))
            task_id = sample.get("id", "unknown")
            for scorer_name, score_data in sample.get("scores", {}).items():
                scores.setdefault(scorer_name, []).append(score_data.get("value"))

        results.append({"model": model, "scores": scores, "file": f})
```

## Step 5: Format Report

Create a results table:

```markdown
## Evaluation Results

| Model | Provider | Overall Score | Tasks Scored | Notes |
|-------|----------|--------------|--------------|-------|
| gpt-4.1 | Azure OpenAI | 0.XX | N/M | ... |
| gpt-4 | OpenAI | 0.XX | N/M | ... |

### Per-Task Breakdown

| Task | gpt-4.1 | gpt-4 |
|------|---------|-------|
| task_1 | 1.0 | 1.0 |
| task_2 | 0.0 | 1.0 |

### Configuration
- Date: YYYY-MM-DD
- Domain: <domain>
- Samples per task: N
- Agent: react (default)
```

## Step 6: Save Report

Save the report to `agent_artifacts/<domain>/evalreport/REPORT.md`.
