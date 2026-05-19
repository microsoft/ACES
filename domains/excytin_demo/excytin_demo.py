# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Excytin Demo - Incident Response domain for Inspect AI.

This module exposes the Excytin Demo domain as an Inspect AI task that can be
evaluated with commands like:

    inspect eval external/saber/domains/excytin_demo --model openai/gpt-4
    inspect eval external/saber/domains/excytin_demo -T task_filter=incident_5_* --model anthropic/claude-3-opus
"""

from inspect_ai import Task, task

from saber.task import create_task


@task
def excytin_demo(**kwargs: str | None) -> Task:
    """Excytin Demo - Incident Response domain.

    Cybersecurity incident response benchmark with database forensics and SQL analysis.

    Args:
        **kwargs: Keyword arguments forwarded to ``create_task``
            (e.g., task_filter, agent, rebuild, run_preflight,
            keep_permanent, persona_file).

    Returns:
        Fully configured inspect_ai Task.
    """
    return create_task(**kwargs)
