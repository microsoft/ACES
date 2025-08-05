"""
Network Investigation Agent for SABER Framework

A simple security agent for network investigation tasks that follows
SABER's zero-friction design principles.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class NetworkInvestigationAgent:
    """
    Simple security agent for network investigation tasks.

    This agent follows SABER's zero-friction design - it just needs:
    - A method that takes text input and returns text output
    - Optional reset method for state management
    """

    def __init__(self):
        """Initialize the agent with investigation state."""
        self.investigation_results: List[Dict[str, Any]] = []
        self.step_count = 0
        self.commands_used = set()

    def reset(self) -> None:
        """Reset agent state for a new investigation."""
        self.investigation_results.clear()
        self.step_count = 0
        self.commands_used.clear()
        logger.info("NetworkInvestigationAgent reset for new investigation")

    def process(self, prompt: str) -> str:
        """
        Process the investigation prompt and return the next command.

        This is the main method that SABER will call with task prompts.
        The agent analyzes the prompt and decides on the next investigation step.
        """
        self.step_count += 1
        logger.info(f"Processing step {self.step_count}")

        # Simple command selection strategy
        command = self._decide_next_command(prompt)

        # Track command usage
        self.commands_used.add(command)

        logger.info(f"Selected command: {command}")
        return command

    def _decide_next_command(self, prompt: str) -> str:
        """
        Decide the next command based on the current prompt and investigation state.

        This implements a simple strategy for network investigation:
        1. Start with network connections (netstat/ss)
        2. Check running processes (ps)
        3. Examine network-process relationships (lsof)
        4. Investigate suspicious files (file/strings)
        """
        prompt_lower = prompt.lower()

        # If we see output from previous commands, analyze it
        if "established" in prompt_lower or "listen" in prompt_lower:
            self._analyze_network_output(prompt)

        if any(proc in prompt_lower for proc in ["pid", "process", "cmd"]):
            self._analyze_process_output(prompt)

        # Command selection priority
        available_commands = [
            "netstat -tulpn",  # Network connections with processes
            "ss -tulpn",       # Alternative network connections
            "ps aux",          # Running processes
            "lsof -i",         # Network files and processes
        ]

        # Select first unused command
        for cmd in available_commands:
            base_cmd = cmd.split()[0]  # Get base command name
            if base_cmd not in self.commands_used:
                return cmd

        # If we've used all commands, provide analysis
        return self._generate_analysis()

    def _analyze_network_output(self, output: str) -> None:
        """Analyze network command output for insights."""
        findings = []

        if "443" in output:
            findings.append("HTTPS connections detected")
        if "22" in output:
            findings.append("SSH connections found")
        if "suspicious" in output.lower():
            findings.append("Suspicious network activity detected")

        if findings:
            self.investigation_results.append({
                "type": "network_analysis",
                "findings": findings,
                "step": self.step_count
            })

    def _analyze_process_output(self, output: str) -> None:
        """Analyze process command output for insights."""
        findings = []

        if any(tool in output.lower() for tool in ["nc", "ncat", "socat"]):
            findings.append("Network tools detected in running processes")
        if "suspicious" in output.lower():
            findings.append("Suspicious process names detected")
        if "/tmp/" in output:
            findings.append("Processes running from /tmp directory")

        if findings:
            self.investigation_results.append({
                "type": "process_analysis",
                "findings": findings,
                "step": self.step_count
            })

    def _generate_analysis(self) -> str:
        """Generate final analysis based on investigation results."""
        if not self.investigation_results:
            return "Investigation complete. No suspicious activity detected."

        analysis = ["=== NETWORK INVESTIGATION ANALYSIS ==="]

        for result in self.investigation_results:
            analysis.append(f"\n{result['type'].upper()} (Step {result['step']}):")
            for finding in result['findings']:
                analysis.append(f"  • {finding}")

        analysis.append(f"\nTotal investigation steps: {self.step_count}")
        analysis.append("Investigation complete.")

        return "\n".join(analysis)
