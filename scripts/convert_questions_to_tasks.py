#!/usr/bin/env python3
"""
Script to convert questions from JSON file to tasks.yaml format.
"""

import json
import yaml
from pathlib import Path

def load_json_questions(json_file_path):
    """Load questions from JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Filter out empty entries
    questions = [q for q in data if q and 'question' in q and 'context' in q and 'answer' in q]
    return questions

def load_existing_tasks(yaml_file_path):
    """Load existing tasks.yaml to get the template structure."""
    with open(yaml_file_path, 'r') as f:
        return yaml.safe_load(f)

def create_task_from_question(question_data, task_index, template_task):
    """Convert a single question to task format."""
    task_id = f"incident_5_task_{task_index + 1}"
    
    # Create base task structure from template
    task = {
        'task_id': task_id,
        'title': task_id,
        'description': question_data['question'],
        'prompt_template_file': template_task['prompt_template_file'],
        'sandbox_environment': template_task['sandbox_environment'],
        'initial_context': {
            'incident_context': question_data['context'],
            'question': question_data['question'],
            'database_connection': template_task['initial_context']['database_connection']
        },
        'evaluation_config': {
            'strategy': 'static',
            'criteria': {
                'expected_answers': [question_data['answer']]
            },
            'scoring': {
                'max_score': 1.0
            }
        }
    }
    
    # Create subtasks from solution
    subtasks = []
    if 'solution' in question_data and question_data['solution']:
        for i, solution_step in enumerate(question_data['solution']):
            subtask = {
                'subtask_id': f"checkpoint_{i + 1}",
                'title': f"Checkpoint {i + 1}",
                'description': solution_step,
                'objective': "Identify key details related to the potential compromise that might help in solving the main task."
            }
            subtasks.append(subtask)
    
    if subtasks:
        task['subtasks'] = subtasks
    
    return task

def create_tasks_yaml(questions, template_yaml):
    """Create new tasks.yaml structure with converted questions."""
    # Create new tasks list
    new_tasks = []
    
    # Get template task for structure reference
    template_task = template_yaml['tasks'][0] if template_yaml['tasks'] else None
    
    if not template_task:
        raise ValueError("No template task found in existing tasks.yaml")
    
    # Convert each question to a task
    for i, question in enumerate(questions):
        task = create_task_from_question(question, i, template_task)
        new_tasks.append(task)
    
    # Create new yaml structure
    new_yaml = {
        'domain': template_yaml['domain'],
        'permanent_environment': template_yaml['permanent_environment'],
        'global_defaults': template_yaml['global_defaults'],
        'tasks': new_tasks
    }
    
    return new_yaml

def main():
    # File paths
    json_file = "/home/amudgerikar/repos/SABER/domains/excytin_demo/server/data/questions/test/incident_5_qa_incident_o1-ga_c42.json"
    yaml_file = "/home/amudgerikar/repos/SABER/domains/excytin_demo/server/config/tasks.yaml"
    output_file = "/home/amudgerikar/repos/SABER/domains/excytin_demo/server/config/tasks_generated.yaml"
    
    # Load data
    print("Loading JSON questions...")
    questions = load_json_questions(json_file)
    print(f"Found {len(questions)} valid questions")
    
    print("Loading existing tasks.yaml template...")
    template_yaml = load_existing_tasks(yaml_file)
    
    # Convert questions to tasks
    print("Converting questions to tasks...")
    new_tasks_yaml = create_tasks_yaml(questions, template_yaml)
    
    # Write output
    print(f"Writing output to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(new_tasks_yaml, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Successfully generated {len(new_tasks_yaml['tasks'])} tasks")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()
