# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for parse_bridge_stderr() in bridge_utils."""

from saber.agents.bridge_utils import parse_bridge_stderr


class TestParseBridgeStderr:
    """Unit tests for structured bridge error parsing."""

    def test_empty_stderr_returns_empty(self) -> None:
        assert parse_bridge_stderr("") == ""

    def test_clean_stderr_returns_empty(self) -> None:
        stderr = "Starting bridge proxy on port 3000\nReady.\n"
        assert parse_bridge_stderr(stderr) == ""

    def test_bad_request_error_unknown_parameter(self) -> None:
        stderr = (
            "Error calling method generate_responses: \n"
            "Request:\n"
            '{"model": "gpt-5", "input": [{"role": "user"}]}\n'
            "Traceback (most recent call last):\n"
            "  File \"/app/service.py\", line 42, in handle\n"
            "BadRequestError('Error code: 400 - "
            '{\"error\": {\"message\": \"Unknown parameter: '
            "'input[3].namespace'\", \"type\": \"invalid_request_error\"}}'\")\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "BadRequestError" in result
        assert "Unknown parameter" in result

    def test_bad_request_error_extracts_json_detail(self) -> None:
        stderr = (
            "BadRequestError('Error code: 400 - "
            '{\"error\": {\"message\": \"max_tokens too large\", '
            '"type\": \"invalid_request_error\"}}'
            "')\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "BadRequestError" in result
        assert "400" in result

    def test_capi_error(self) -> None:
        stderr = (
            "CAPIError: Connection refused: could not reach endpoint "
            "at localhost:3000\n"
            "Traceback follows.\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "CAPIError" in result
        assert "Connection refused" in result

    def test_error_calling_method(self) -> None:
        stderr = (
            "Error calling method generate_responses: something went wrong\n"
            "Traceback (most recent call last):\n"
            "  ...\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "ErrorCallingMethod" in result
        assert "generate_responses" in result

    def test_runtime_error(self) -> None:
        stderr = (
            "RuntimeError: Bridge proxy crashed unexpectedly\n"
            "Traceback (most recent call last):\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "RuntimeError" in result
        assert "Bridge proxy crashed" in result

    def test_unknown_parameter_keyword(self) -> None:
        stderr = "blah blah Unknown parameter: 'input[3].namespace' blah\n"
        result = parse_bridge_stderr(stderr)
        assert "Unknown parameter" in result

    def test_invalid_request_error_keyword(self) -> None:
        stderr = 'some output with "type": "invalid_request_error" in it\n'
        result = parse_bridge_stderr(stderr)
        assert "invalid_request_error" in result

    def test_multiple_patterns_in_one_stderr(self) -> None:
        stderr = (
            "CAPIError: Connection timeout at localhost:3000\n"
            "\n"
            "BadRequestError('Error code: 400 - "
            '{\"error\": {\"message\": \"bad param\"}}'
            "')\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "CAPIError" in result
        assert "BadRequestError" in result

    def test_large_stderr_is_handled_efficiently(self) -> None:
        """Ensure parsing works on very large stderr without issues."""
        # Simulate a 15k-char request body dump with an error at the end
        large_body = "x" * 15000
        stderr = (
            f"Error calling method generate_responses: \nRequest:\n{large_body}\n"
            "BadRequestError('Error code: 400 - "
            '{\"error\": {\"message\": \"Unknown parameter\"}}'
            "')\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "BadRequestError" in result

    def test_connection_error_keyword_dedup(self) -> None:
        """ConnectionError pattern and 'Connection error' keyword should not duplicate."""
        stderr = "Connection error: failed to connect to model proxy\n"
        result = parse_bridge_stderr(stderr)
        assert "ConnectionError" in result
        # Should NOT produce a duplicate [Keyword] line
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 1, f"Expected 1 finding, got {len(lines)}: {result}"

    def test_unexpected_error_during_proxy_call(self) -> None:
        stderr = (
            "Unexpected error during model proxy call: "
            "timeout after 30s\n"
        )
        result = parse_bridge_stderr(stderr)
        assert "UnexpectedProxyError" in result
        assert "timeout after 30s" in result

    def test_runtime_error_no_newline_bleed(self) -> None:
        """RuntimeError detail should not bleed across newlines."""
        stderr = (
            "RuntimeError: something broke\n"
            "BadRequestError('Error code: 400 - {}')"
        )
        result = parse_bridge_stderr(stderr)
        # Each finding should be a single line
        for line in result.splitlines():
            assert line.startswith("["), f"Unlabeled line found: {line}"

    def test_error_calling_method_no_request_body_noise(self) -> None:
        """ErrorCallingMethod should capture method name, not the request body."""
        stderr = (
            "Error calling method generate_responses: \n"
            "Request:\n"
            '{"model": "gpt-5", "input": [{"really_long_json": "' + "x" * 1000 + '"}]}\n'
            "\n"
            "BadRequestError('Error code: 400')"
        )
        result = parse_bridge_stderr(stderr)
        method_finding = [f for f in result.splitlines() if "ErrorCallingMethod" in f]
        assert len(method_finding) == 1
        # Should contain the method name but not the request body
        assert "generate_responses" in method_finding[0]
        assert "really_long_json" not in method_finding[0]
