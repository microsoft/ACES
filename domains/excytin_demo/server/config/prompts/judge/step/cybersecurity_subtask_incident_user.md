## Incident Context
**Domain:** {{ domain }}
**Task:** {{ task_id }}
**Incident Question:** {{ question }}

## Agent Execution Trace ({{ episode.get_step_count() }} steps)

{% for step in episode.steps %}
**[{{ step.step_number }}]** {{ step.action.tool_name }}
{% if step.action.reasoning %}
Agent Reasoning: {{ step.action.reasoning | truncate(600) }}
{% endif %}
{% if step.action.assistant_message %}
Agent Message: {{ step.action.assistant_message | truncate(800) }}
{% endif %}
Input: {{ step.action.parameters | string | truncate(200) }}
Output: {{ step.response | string | truncate(800) }}
{% if step.done %}>>> Episode ended here <<<{% endif %}

{% endfor %}

---
