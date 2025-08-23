"""
Tests for PromptBuilder functionality.

Tests the component responsible for building prompts from task and policy information.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest

from saber.api_models import PolicyInfo, StepResponse, TaskInfo
from saber.client.prompt_builder import PromptBuilder


@pytest.fixture
def sample_task_info():
    """Sample TaskInfo for testing."""
    return TaskInfo(
        task_id="malware-analysis-001",
        title="Malware Sample Analysis",
        description="Analyze the suspicious file located at /tmp/sample.exe to determine if it's malicious",
        current_subtask="initial-inspection",
    )


@pytest.fixture
def sample_policy_info():
    """Sample PolicyInfo for testing."""
    return PolicyInfo(
        domain="malware-analysis",
        available_commands=["file", "strings", "hexdump", "ls", "cat", "grep"],
        guidelines="Use multiple analysis techniques. Start with basic file inspection, then proceed to strings analysis and hex dumps. Look for suspicious patterns, URLs, and embedded files.",
        constraints=["No network access allowed", "Read-only filesystem", "No execution of suspicious files"],
    )


@pytest.fixture
def sample_step_responses():
    """Sample step responses for testing."""
    return [
        StepResponse(
            success=True,
            output="sample.exe: PE32 executable (GUI) Intel 80386, for MS Windows",
            done=False,
            error=None,
            info={"command": "file /tmp/sample.exe"},
        ),
        StepResponse(
            success=True,
            output="http://malicious-domain.com/payload\nC:\\Users\\victim\\Desktop\nCreateProcessA",
            done=False,
            error=None,
            info={"command": "strings /tmp/sample.exe | head -10"},
        ),
    ]


class TestPromptBuilder:
    """Test cases for PromptBuilder."""

    def test_init(self):
        """Test PromptBuilder initialization."""
        builder = PromptBuilder()
        assert builder is not None

    @pytest.mark.asyncio
    async def test_build_initial_prompt_basic(self, sample_task_info, sample_policy_info):
        """Test building initial prompt with basic information."""
        builder = PromptBuilder()

        prompt = builder.build_initial_prompt(sample_task_info, sample_policy_info)

        # Check that prompt contains key information
        assert "Malware Sample Analysis" in prompt
        assert "Analyze the suspicious file" in prompt
        assert "file" in prompt  # Available command
        assert "strings" in prompt  # Available command    @pytest.mark.asyncio

    def test_build_initial_prompt_formatting(self, sample_task_info, sample_policy_info):
        """Test that initial prompt is well-formatted."""
        builder = PromptBuilder()

        prompt = builder.build_initial_prompt(sample_task_info, sample_policy_info)

        # Check for proper sections
        assert "Task:" in prompt
        assert "Available commands:" in prompt
        assert "Constraints:" in prompt
        assert "Guidelines:" in prompt

        # Check formatting
        lines = prompt.split("\n")
        assert len(lines) > 5  # Should be multi-line

        # Should have clear structure
        task_section_found = False
        commands_section_found = False
        for line in lines:
            if "Task:" in line:
                task_section_found = True
            if "Available commands:" in line:
                commands_section_found = True

        assert task_section_found
        assert commands_section_found

    @pytest.mark.asyncio
    async def test_build_initial_prompt_minimal_info(self):
        """Test building initial prompt with minimal information."""
        builder = PromptBuilder()

        minimal_task = TaskInfo(task_id="test-task", title="Test", description="", current_subtask="")

        minimal_policy = PolicyInfo(domain="test", available_commands=[], guidelines="", constraints=[])

        prompt = builder.build_initial_prompt(minimal_task, minimal_policy)

        # Should still have basic structure
        assert "Task:" in prompt
        assert "Test" in prompt

        # Should handle empty lists gracefully
        assert "Available commands:" in prompt

    @pytest.mark.asyncio
    async def test_build_step_prompt_basic(self, sample_step_responses):
        """Test building step prompt with command history."""
        builder = PromptBuilder()

        # Extract output from the first response
        last_output = sample_step_responses[0].output
        previous_command = sample_step_responses[0].info.get("command", "")

        prompt = builder.build_step_prompt(last_output, previous_command, sample_step_responses[0])

        # Should contain command and output
        assert "file /tmp/sample.exe" in prompt
        assert "PE32 executable" in prompt

        # Should have clear structure
        assert "Previous command:" in prompt or "Output:" in prompt

    @pytest.mark.asyncio
    async def test_build_step_prompt_empty_history(self):
        """Test building step prompt with no command history."""
        builder = PromptBuilder()

        prompt = builder.build_step_prompt("", None, None)

        # Should handle empty history gracefully
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Should indicate no output
        assert "(No output)" in prompt or "no output" in prompt.lower()

    @pytest.mark.asyncio
    async def test_build_step_prompt_with_errors(self):
        """Test building step prompt with failed commands."""
        builder = PromptBuilder()

        step_response = StepResponse(
            success=False, output="", done=False, error="Permission denied", info={"command": "cat /etc/shadow"}
        )

        prompt = builder.build_step_prompt("", "cat /etc/shadow", step_response)

        # Should include command and error
        assert "cat /etc/shadow" in prompt
        assert "Permission denied" in prompt or "Error:" in prompt

    @pytest.mark.asyncio
    async def test_build_step_prompt_truncation(self):
        """Test that step prompt handles long outputs appropriately."""
        builder = PromptBuilder()

        # Create a response with very long output
        long_output = "x" * 10000  # 10KB of text

        step_response = StepResponse(
            success=True, output=long_output, done=False, error=None, info={"command": "cat large_file.txt"}
        )

        prompt = builder.build_step_prompt(long_output, "cat large_file.txt", step_response)

        # Prompt should be reasonable length (implementation-dependent)
        # This test ensures we don't create massive prompts
        assert len(prompt) < 50000  # Reasonable upper bound
        assert "cat large_file.txt" in prompt

    @pytest.mark.asyncio
    async def test_build_step_prompt_command_extraction(self):
        """Test building step prompt with command information."""
        builder = PromptBuilder()

        step_response = StepResponse(success=True, output="output1", done=False, error=None, info={"command": "ls -la"})

        prompt = builder.build_step_prompt("output1", "ls -la", step_response)

        # Should handle command and output
        assert "ls -la" in prompt
        assert "output1" in prompt


class TestPromptBuilderCustomization:
    """Test customization and edge cases for PromptBuilder."""

    @pytest.mark.asyncio
    async def test_custom_formatting(self, sample_task_info, sample_policy_info):
        """Test that prompts use consistent formatting."""
        builder = PromptBuilder()

        prompt = builder.build_initial_prompt(sample_task_info, sample_policy_info)

        # Should use consistent section headers (exclude final instruction line)
        lines = prompt.split("\n")
        section_headers = [
            line
            for line in lines
            if ":" in line and not line.strip().startswith("  ") and not line.startswith("Please provide")
        ]

        # Should have multiple clear sections
        assert len(section_headers) >= 2

        # Each section should be followed by content
        for header in section_headers:
            header_line_num = lines.index(header)
            # Should have content after header (not just empty lines)
            has_content = False
            for j in range(header_line_num + 1, min(header_line_num + 5, len(lines))):
                if j < len(lines) and lines[j].strip():
                    has_content = True
                    break
            assert has_content, f"Section {header} should have content"

    @pytest.mark.asyncio
    async def test_command_list_formatting(self, sample_task_info, sample_policy_info):
        """Test formatting of available commands list."""
        builder = PromptBuilder()

        prompt = builder.build_initial_prompt(sample_task_info, sample_policy_info)

        # Commands should be listed in a readable format
        for command in sample_policy_info.available_commands:
            assert command in prompt

        # Should not have excessive punctuation or poor formatting
        command_section = ""
        in_command_section = False
        for line in prompt.split("\n"):
            if "Available commands:" in line:
                in_command_section = True
            elif in_command_section and line.strip() and ":" in line:
                break  # Next section
            elif in_command_section:
                command_section += line + "\n"

        # Command section should be readable
        assert len(command_section.strip()) > 0

    @pytest.mark.asyncio
    async def test_constraint_formatting(self, sample_task_info, sample_policy_info):
        """Test formatting of constraints list."""
        builder = PromptBuilder()

        prompt = builder.build_initial_prompt(sample_task_info, sample_policy_info)

        # Constraints should be clearly listed
        for constraint in sample_policy_info.constraints:
            assert constraint in prompt

        # Should have clear constraint section
        assert "Constraints:" in prompt


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_malware_analysis_workflow(self):
        """Test prompt building for a complete malware analysis workflow."""
        builder = PromptBuilder()

        # Initial task setup
        task = TaskInfo(
            task_id="malware-001",
            title="Suspicious Email Attachment Analysis",
            description="Analyze attachment.zip from suspicious email for potential malware",
            current_subtask="file-identification",
        )

        policy = PolicyInfo(
            domain="malware-analysis",
            available_commands=["file", "unzip", "strings", "hexdump", "md5sum", "sha256sum"],
            guidelines="Follow standard malware analysis procedures. Start with static analysis.",
            constraints=["Sandboxed environment only", "No network connectivity"],
        )

        # Build initial prompt
        initial_prompt = builder.build_initial_prompt(task, policy)

        # Verify malware-specific content
        assert "malware" in initial_prompt.lower()
        assert "attachment.zip" in initial_prompt
        assert "static analysis" in initial_prompt
        assert "sandboxed" in initial_prompt.lower()

        # Simulate some analysis steps
        steps = [
            StepResponse(
                success=True,
                output="attachment.zip: Zip archive data, at least v2.0 to extract",
                done=False,
                error=None,
                info={"command": "file attachment.zip"},
            ),
            StepResponse(
                success=True,
                output="  Length      Date    Time    Name\n---------  ---------- -----   ----\n     2048  2023-12-01 10:30   invoice.exe",
                done=False,
                error=None,
                info={"command": "unzip -l attachment.zip"},
            ),
        ]

        # Build step prompt
        step_prompt = builder.build_step_prompt(
            steps[-1].output,  # Use the last output as a string
            "unzip -l attachment.zip",  # Previous command
            steps[-1],  # Full response
        )

        # Verify step progression
        assert "unzip -l attachment.zip" in step_prompt  # The previous command shown
        assert "invoice.exe" in step_prompt
        assert "Length" in step_prompt  # Part of the output

    @pytest.mark.asyncio
    async def test_network_troubleshooting_workflow(self):
        """Test prompt building for network troubleshooting scenario."""
        builder = PromptBuilder()

        task = TaskInfo(
            task_id="network-001",
            title="Network Connectivity Issues",
            description="Diagnose network connectivity problems on the target system",
            current_subtask="initial-assessment",
        )

        policy = PolicyInfo(
            domain="network-diagnostics",
            available_commands=["ping", "traceroute", "netstat", "ss", "ip", "dig"],
            guidelines="Start with basic connectivity tests, then examine routing and DNS",
            constraints=["Limited to diagnostic commands only"],
        )

        initial_prompt = builder.build_initial_prompt(task, policy)

        # Verify network-specific content
        assert "network" in initial_prompt.lower()
        assert "connectivity" in initial_prompt.lower()
        assert "ping" in initial_prompt
        assert "routing" in initial_prompt.lower()

        # Simulate diagnostic steps
        steps = [
            StepResponse(
                success=False,
                output="ping: connect: Network is unreachable",
                done=False,
                error="Network unreachable",
                info={"command": "ping 8.8.8.8"},
            )
        ]

        step_prompt = builder.build_step_prompt(
            steps[0].output, "ping 8.8.8.8", steps[0]  # Use the output as a string  # Previous command  # Full response
        )

        # Should handle network errors appropriately
        assert "ping 8.8.8.8" in step_prompt
        assert "unreachable" in step_prompt.lower()

    @pytest.mark.asyncio
    async def test_empty_and_none_handling(self):
        """Test handling of None and empty values."""
        builder = PromptBuilder()

        # Test with None values (shouldn't happen in practice, but good to be safe)
        task = TaskInfo(task_id="test", title="", description="", current_subtask="")

        policy = PolicyInfo(domain="", available_commands=[], guidelines="", constraints=[])

        # Should not raise exceptions
        prompt = builder.build_initial_prompt(task, policy)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test with empty step responses
        step_prompt = builder.build_step_prompt("")  # Empty output
        assert isinstance(step_prompt, str)
