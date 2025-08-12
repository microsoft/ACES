"""
SABER Client - Prompt building utilities.

Builds prompts for agents from server data. Formats task information,
command outputs, and policy data into clear prompts that agents can process.
"""

from typing import Optional

from ..api_models import EpisodeInfo, PolicyInfo, StepResponse, TaskInfo


class PromptBuilder:
    """
    Builds formatted prompts for agents from server data.

    Creates clear, structured prompts that provide agents with all necessary
    context while hiding server implementation details.
    """

    @staticmethod
    def build_initial_prompt(
        task_info: TaskInfo, policy_info: PolicyInfo, episode_info: Optional[EpisodeInfo] = None
    ) -> str:
        """
        Build the initial prompt for starting a task.

        Args:
            task_info: Current task information
            policy_info: Domain policy and available commands
            episode_info: Optional episode information

        Returns:
            str: Formatted initial prompt for the agent
        """
        prompt_parts = []

        # Task information
        prompt_parts.append(f"Task: {task_info.title}")
        prompt_parts.append(f"Description: {task_info.description}")

        prompt_parts.append("")

        prompt_parts.append("Available commands:")
        commands_list = ", ".join(policy_info.available_commands)
        prompt_parts.append(f"  {commands_list}")
        prompt_parts.append("")

        if policy_info.guidelines:
            prompt_parts.append("Guidelines:")
            prompt_parts.append(f"  {policy_info.guidelines}")
            prompt_parts.append("")

        if policy_info.constraints:
            prompt_parts.append("Constraints:")
            for constraint in policy_info.constraints:
                prompt_parts.append(f"  - {constraint}")
            prompt_parts.append("")

        # Instructions
        prompt_parts.append("Instructions:")
        prompt_parts.append("  - Analyze the situation using the available commands")
        prompt_parts.append("  - Provide one command at a time")
        prompt_parts.append("  - Commands should be specific and actionable")
        prompt_parts.append("  - Use proper command syntax and parameters")
        prompt_parts.append("")
        prompt_parts.append("Please provide your first command:")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_step_prompt(
        last_output: str, previous_command: Optional[str] = None, step_response: Optional[StepResponse] = None
    ) -> str:
        """
        Build a follow-up prompt with command output.

        Args:
            last_output: Output from the last command execution
            previous_command: The command that was executed (optional)
            step_response: Full step response from server (optional)

        Returns:
            str: Formatted prompt with command output
        """
        prompt_parts = []

        if previous_command:
            prompt_parts.append(f"Previous command: {previous_command}")
            prompt_parts.append("")

        # Output
        if last_output.strip():
            prompt_parts.append("Output:")
            prompt_parts.append(last_output)
        else:
            prompt_parts.append("(No output)")

        prompt_parts.append("")

        # Error information if available
        if step_response and step_response.error:
            prompt_parts.append("Error:")
            prompt_parts.append(step_response.error)
            prompt_parts.append("")

        # Next instruction
        if step_response and step_response.done:
            prompt_parts.append("Task completed! Analysis finished.")
        else:
            prompt_parts.append("Based on this output, what is your next command?")
            prompt_parts.append("(Provide just the command, no explanation needed)")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_error_prompt(error_message: str, previous_command: Optional[str] = None) -> str:
        """
        Build a prompt for when an error occurs.

        Args:
            error_message: The error that occurred
            previous_command: The command that caused the error

        Returns:
            str: Formatted error prompt
        """
        prompt_parts = []

        if previous_command:
            prompt_parts.append(f"Failed command: {previous_command}")
            prompt_parts.append("")

        prompt_parts.append("Error:")
        prompt_parts.append(error_message)
        prompt_parts.append("")
        prompt_parts.append("Please try a different command or approach.")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_completion_prompt(final_output: str) -> str:
        """
        Build a completion prompt when the task is done.

        Args:
            final_output: Final output from the completed task

        Returns:
            str: Formatted completion prompt
        """
        prompt_parts = []

        if final_output.strip():
            prompt_parts.append("Final result:")
            prompt_parts.append(final_output)
            prompt_parts.append("")

        prompt_parts.append("Analysis completed successfully!")

        return "\n".join(prompt_parts)
