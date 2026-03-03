## Incident Context
**Domain:** {{ domain }}
**Task:** {{ task_id }}
**Incident Question:** {{ question }}

## Expected Finding
**Golden Answer:** {{ description }}

## Agent Step-by-Step Execution Analysis
**Final Submission:** {{ submission }}
**Total Steps:** {{ episode.get_step_count() }}

### Complete Step-by-Step Investigation:
{% for step in episode.steps %}
**Step {{ step.step_number }}**
- Tool: {{ step.tool_name }}
- Input: {{ step.tool_input | string | truncate(150) }}
- Output: {{ step.output | string | truncate(1500) }}

{% endfor %}

### Database Query Analysis:
{% for step in episode.steps %}
{% if step.tool_name == "mysql" or "mysql" in (step.tool_input | string) %}
**Database Query Step {{ step.step_number }}:**
- Input: {{ step.tool_input | string | truncate(200) }}
- Result: {{ step.output | string | truncate(350) }}
{% endif %}
{% endfor %}

## Evaluation Task

Analyze each step above against the checkpoint objectives. Identify which steps successfully completed which objectives (checkpoints or main task). Consider the technical accuracy of findings, investigation methodology, and logical progression through the security incident analysis.

Provide your step evaluation in the required STEP_EVALUATIONS format.
