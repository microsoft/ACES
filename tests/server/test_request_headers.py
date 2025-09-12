"""
Unit tests for RequestHeaders model and OrchestrationEnvironment enum.

Tests header parsing, validation, and orchestration environment functionality.
"""

import pytest

from saber.models.mcp import OrchestrationEnvironment, RequestHeaders


class TestOrchestrationEnvironment:
    """Test OrchestrationEnvironment enum functionality."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert OrchestrationEnvironment.INSPECT == "inspect"
        assert OrchestrationEnvironment.STANDALONE == "standalone"

    def test_string_representation(self):
        """Test string representation of enum values."""
        assert str(OrchestrationEnvironment.INSPECT) == "inspect"
        assert str(OrchestrationEnvironment.STANDALONE) == "standalone"

    def test_is_valid_method(self):
        """Test is_valid class method."""
        assert OrchestrationEnvironment.is_valid("inspect") is True
        assert OrchestrationEnvironment.is_valid("standalone") is True
        assert OrchestrationEnvironment.is_valid("invalid") is False
        assert OrchestrationEnvironment.is_valid("") is False
        assert OrchestrationEnvironment.is_valid(None) is False

    def test_get_valid_values_method(self):
        """Test get_valid_values class method."""
        valid_values = OrchestrationEnvironment.get_valid_values()
        assert valid_values == ["inspect", "standalone"]
        assert isinstance(valid_values, list)

    def test_enum_comparison(self):
        """Test enum comparison functionality."""
        inspect1 = OrchestrationEnvironment.INSPECT
        inspect2 = OrchestrationEnvironment("inspect")
        standalone = OrchestrationEnvironment.STANDALONE

        assert inspect1 == inspect2
        assert inspect1 != standalone
        assert inspect1 == "inspect"
        assert standalone == "standalone"


class TestRequestHeaders:
    """Test RequestHeaders dataclass functionality."""

    def test_init_with_all_values(self):
        """Test RequestHeaders initialization with all values."""
        headers = RequestHeaders(
            session_id="test-session",
            episode_id="test-episode",
            orchestration_env=OrchestrationEnvironment.INSPECT,
            task_id="test-task",
            client_id="test-client"
        )

        assert headers.session_id == "test-session"
        assert headers.episode_id == "test-episode"
        assert headers.orchestration_env == OrchestrationEnvironment.INSPECT
        assert headers.task_id == "test-task"
        assert headers.client_id == "test-client"

    def test_init_with_minimal_values(self):
        """Test RequestHeaders initialization with minimal required values."""
        headers = RequestHeaders(
            session_id=None,
            episode_id=None,
            orchestration_env=OrchestrationEnvironment.STANDALONE
        )

        assert headers.session_id is None
        assert headers.episode_id is None
        assert headers.orchestration_env == OrchestrationEnvironment.STANDALONE
        assert headers.task_id is None
        assert headers.client_id is None

    def test_post_init_validation(self):
        """Test that post_init validation works correctly."""
        # Valid case - should not raise
        RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        # Invalid case - orchestration_env is None should raise
        with pytest.raises(ValueError, match="orchestration_env is mandatory but was None"):
            # This would require manually setting to None after init, but dataclass prevents this
            # So we test the logic by creating a headers object and manually setting to None
            headers = RequestHeaders(
                session_id="test",
                episode_id="test",
                orchestration_env=OrchestrationEnvironment.INSPECT
            )
            headers.orchestration_env = None
            headers.__post_init__()

    def test_has_session_context(self):
        """Test has_session_context property."""
        # With session context
        headers_with_session = RequestHeaders(
            session_id="test-session",
            episode_id=None,
            orchestration_env=OrchestrationEnvironment.INSPECT
        )
        assert headers_with_session.has_session_context is True

        # Without session context
        headers_without_session = RequestHeaders(
            session_id=None,
            episode_id="test-episode",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )
        assert headers_without_session.has_session_context is False

    def test_has_episode_context(self):
        """Test has_episode_context property."""
        # With episode context
        headers_with_episode = RequestHeaders(
            session_id=None,
            episode_id="test-episode",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )
        assert headers_with_episode.has_episode_context is True

        # Without episode context
        headers_without_episode = RequestHeaders(
            session_id="test-session",
            episode_id=None,
            orchestration_env=OrchestrationEnvironment.INSPECT
        )
        assert headers_without_episode.has_episode_context is False

    def test_context_summary_full_context(self):
        """Test context_summary with full context."""
        headers = RequestHeaders(
            session_id="sess-123",
            episode_id="ep-456",
            orchestration_env=OrchestrationEnvironment.STANDALONE,
            task_id="task-789",
            client_id="client-abc"
        )

        summary = headers.context_summary
        assert "session:sess-123" in summary
        assert "episode:ep-456" in summary
        assert "task:task-789" in summary
        assert "orchestration:standalone" in summary
        assert summary.startswith("[")
        assert summary.endswith("]")

    def test_context_summary_minimal_context(self):
        """Test context_summary with minimal context."""
        headers = RequestHeaders(
            session_id=None,
            episode_id=None,
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        summary = headers.context_summary
        assert "orchestration:inspect" in summary
        assert "session:" not in summary
        assert "episode:" not in summary
        assert "task:" not in summary
        assert summary == "[orchestration:inspect]"

    def test_context_summary_partial_context(self):
        """Test context_summary with partial context."""
        headers = RequestHeaders(
            session_id="sess-123",
            episode_id=None,
            orchestration_env=OrchestrationEnvironment.STANDALONE,
            task_id="task-789"
        )

        summary = headers.context_summary
        assert "session:sess-123" in summary
        assert "task:task-789" in summary
        assert "orchestration:standalone" in summary
        assert "episode:" not in summary
        assert summary.count(",") == 2  # Three items total

    def test_orchestration_environment_types(self):
        """Test that different orchestration environments work correctly."""
        inspect_headers = RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        standalone_headers = RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.STANDALONE
        )

        assert inspect_headers.orchestration_env == OrchestrationEnvironment.INSPECT
        assert standalone_headers.orchestration_env == OrchestrationEnvironment.STANDALONE
        assert inspect_headers.orchestration_env != standalone_headers.orchestration_env

    def test_dataclass_equality(self):
        """Test that RequestHeaders equality works correctly."""
        headers1 = RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        headers2 = RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        headers3 = RequestHeaders(
            session_id="different",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        assert headers1 == headers2
        assert headers1 != headers3

    def test_dataclass_immutability_expectations(self):
        """Test that RequestHeaders behaves as expected for modification."""
        headers = RequestHeaders(
            session_id="test",
            episode_id="test",
            orchestration_env=OrchestrationEnvironment.INSPECT
        )

        # We can modify fields (dataclass is not frozen by default)
        headers.session_id = "modified"
        assert headers.session_id == "modified"

        # Context properties should update accordingly
        assert "session:modified" in headers.context_summary
