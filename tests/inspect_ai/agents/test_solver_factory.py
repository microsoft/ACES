"""Tests for SABER solver factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.solver import TaskState

from saber.inspect_ai.agents.solver_factory import (
    SABERExecutionContext,
    create_saber_solver,
)
from saber.inspect_ai.constants import InspectStoreKeys
from saber.models.constants import MetadataKeys


@pytest.fixture
def mock_state():
    """Create mock TaskState."""
    state = MagicMock(spec=TaskState)
    state.metadata = {}
    state.store = MagicMock()
    state.store.get = MagicMock(return_value={})
    state.messages = []
    state.output = MagicMock()
    state.output.completion = "test completion"
    return state


@pytest.fixture
def mock_generate():
    """Create mock Generate."""
    return MagicMock()


@pytest.fixture
def complete_metadata():
    """Complete metadata for testing."""
    return {
        MetadataKeys.SESSION_ID: "session-123",
        MetadataKeys.EPISODE_ID: "episode-456",
        MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
        MetadataKeys.TASK_ID: "task-789",
        MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
        MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        MetadataKeys.SUBMIT_PROMPT: "Test submit",
        MetadataKeys.SAMPLE_ID: "sample-1",
    }


class TestCreateSaberSolver:
    """Test create_saber_solver function."""

    @pytest.mark.asyncio
    async def test_solver_with_complete_metadata(self, mock_state, mock_generate, complete_metadata):
        """Test solver execution with complete metadata."""
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        # Create mock agent
        mock_agent = AsyncMock(return_value=mock_state)

        def mock_create_with_prompts(instruction_prompt, assistant_prompt, submit_prompt):
            return mock_agent

        mock_factory = MagicMock(return_value=mock_create_with_prompts)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_model = MagicMock()
                mock_active_model.return_value = mock_model

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var:
                    solver = create_saber_solver("test-agent", mock_factory)
                    result = await solver(mock_state, mock_generate)

        # Verify agent was called
        mock_agent.assert_called_once_with(mock_state)
        assert result == mock_state

    @pytest.mark.asyncio
    async def test_solver_missing_instruction_prompt(self, mock_state, mock_generate):
        """Test solver fails when instruction_prompt is missing."""
        mock_state.metadata = {
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
        }
        mock_state.store.get.return_value = None

        mock_factory = MagicMock()

        solver = create_saber_solver("test-agent", mock_factory)

        with pytest.raises(ValueError, match="Missing 'instruction_prompt'"):
            await solver(mock_state, mock_generate)

    @pytest.mark.asyncio
    async def test_solver_missing_assistant_prompt(self, mock_state, mock_generate):
        """Test solver fails when assistant_prompt is missing."""
        mock_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
        }
        mock_state.store.get.return_value = None

        mock_factory = MagicMock()

        solver = create_saber_solver("test-agent", mock_factory)

        with pytest.raises(ValueError, match="Missing 'assistant_prompt'"):
            await solver(mock_state, mock_generate)

    @pytest.mark.asyncio
    async def test_solver_missing_submit_prompt(self, mock_state, mock_generate):
        """Test solver fails when submit_prompt is missing."""
        mock_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
        }
        mock_state.store.get.return_value = None

        mock_factory = MagicMock()

        solver = create_saber_solver("test-agent", mock_factory)

        with pytest.raises(ValueError, match="Missing 'submit_prompt'"):
            await solver(mock_state, mock_generate)

    @pytest.mark.asyncio
    async def test_solver_with_role_based_model(self, mock_state, mock_generate, complete_metadata):
        """Test solver with role-based model selection."""
        complete_metadata[MetadataKeys.SUB_TASK_ROLE] = "blue"
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        # Mock role config
        mock_role_config = MagicMock()
        mock_role_agent_config = MagicMock()
        mock_role_agent_config.model = "gpt-4"
        mock_role_config.get_config_for_role.return_value = mock_role_agent_config

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_model") as mock_get_model:
            mock_role_model = MagicMock()
            mock_get_model.return_value = mock_role_model

            with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
                mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                    solver = create_saber_solver("test-agent", mock_factory, role_config=mock_role_config)
                    await solver(mock_state, mock_generate)

            # Verify role-specific model was requested
            mock_get_model.assert_called_once_with("gpt-4")
            mock_role_config.get_config_for_role.assert_called_once_with("blue")

    @pytest.mark.asyncio
    async def test_solver_role_model_fallback(self, mock_state, mock_generate, complete_metadata):
        """Test solver falls back to default model when role model fails."""
        complete_metadata[MetadataKeys.SUB_TASK_ROLE] = "red"
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        # Mock role config that raises error
        mock_role_config = MagicMock()
        mock_role_config.get_config_for_role.side_effect = Exception("Config error")

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
            mock_default_model = MagicMock()
            mock_active_model.return_value = mock_default_model

            with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
                mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                    solver = create_saber_solver("test-agent", mock_factory, role_config=mock_role_config)
                    await solver(mock_state, mock_generate)

            # Should have used default model
            mock_active_model.assert_called()

    @pytest.mark.asyncio
    async def test_solver_model_wrapping_for_transcript_sync(self, mock_state, mock_generate, complete_metadata):
        """Test solver wraps model for transcript sync when context is complete."""
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper:
            mock_wrapped = MagicMock()
            mock_wrapper.return_value = mock_wrapped

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_base_model = MagicMock()
                mock_active_model.return_value = mock_base_model

                with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
                    mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

                    with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                        solver = create_saber_solver("test-agent", mock_factory)
                        await solver(mock_state, mock_generate)

            # Verify model was wrapped
            mock_wrapper.assert_called_once_with(
                base_model=mock_base_model,
                session_id="session-123",
                episode_id="episode-456",
                rest_url="http://localhost:8000",
            )

    @pytest.mark.asyncio
    async def test_solver_no_wrapping_without_complete_context(self, mock_state, mock_generate):
        """Test solver doesn't wrap model when context is incomplete."""
        mock_state.metadata = {
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            # Missing session_id, episode_id, domain_slug
        }
        mock_state.store.get.return_value = None

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_wrapper:
            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_base_model = MagicMock()
                mock_active_model.return_value = mock_base_model

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                    solver = create_saber_solver("test-agent", mock_factory)
                    await solver(mock_state, mock_generate)

            # Verify model was NOT wrapped
            mock_wrapper.assert_not_called()

    @pytest.mark.asyncio
    async def test_solver_model_context_restoration(self, mock_state, mock_generate, complete_metadata):
        """Test solver restores previous model context after execution."""
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
            mock_previous_model = MagicMock()
            mock_active_model.return_value = mock_previous_model

            with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
                mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var") as mock_context_var:
                    solver = create_saber_solver("test-agent", mock_factory)
                    await solver(mock_state, mock_generate)

            # Verify model was set and then restored
            assert mock_context_var.set.call_count >= 2  # Set wrapped, then restore

    @pytest.mark.asyncio
    async def test_solver_episode_mapping_not_found(self, mock_state, mock_generate, complete_metadata):
        """Test solver handles missing episode mapping gracefully."""
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {},  # Empty mapping
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
            mock_active_model.return_value = MagicMock()

            with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                # Should not raise, just log warning
                solver = create_saber_solver("test-agent", mock_factory)
                await solver(mock_state, mock_generate)

    @pytest.mark.asyncio
    async def test_solver_domain_not_found(self, mock_state, mock_generate, complete_metadata):
        """Test solver handles domain not found gracefully."""
        mock_state.metadata = complete_metadata.copy()
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = None  # Domain not found

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_active_model.return_value = MagicMock()

                with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                    # Should not raise, just log warning
                    solver = create_saber_solver("test-agent", mock_factory)
                    await solver(mock_state, mock_generate)


class TestSolverFactoryBlockingWrapper:
    """Test solver_factory blocking wrapper creation."""

    @pytest.mark.asyncio
    async def test_creates_blocking_wrapper_when_enabled(self, mock_state, mock_generate):
        """Test that BlockingTranscriptSyncingModelWrapper is created when blocking_config.enabled=True."""
        metadata = {
            MetadataKeys.SESSION_ID: "session-123",
            MetadataKeys.EPISODE_ID: "episode-456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.SAMPLE_ID: "sample-1",
            "blocking_config": {
                "enabled": True,
                "poll_interval": 2.0,
                "max_iterations": 50,
                "timeout": 100.0,
                "skip_first_iteration": True,
            }
        }

        mock_state.metadata = metadata
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_model = MagicMock()
                mock_active_model.return_value = mock_model

                with patch("saber.inspect_ai.agents.solver_factory.BlockingTranscriptSyncingModelWrapper") as mock_blocking_wrapper:
                    mock_blocking_wrapper.return_value = MagicMock()

                    with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                        solver = create_saber_solver("test-agent", mock_factory)
                        await solver(mock_state, mock_generate)

                        # Verify BlockingTranscriptSyncingModelWrapper was created
                        mock_blocking_wrapper.assert_called_once()
                        call_args = mock_blocking_wrapper.call_args
                        assert call_args[1]["base_model"] == mock_model
                        assert call_args[1]["session_id"] == "session-123"
                        assert call_args[1]["episode_id"] == "episode-456"
                        assert call_args[1]["rest_url"] == "http://localhost:8000"
                        assert call_args[1]["poll_interval"] == 2.0
                        assert call_args[1]["max_iterations"] == 50
                        assert call_args[1]["timeout"] == 100.0
                        assert call_args[1]["skip_first_iteration"] is True

    @pytest.mark.asyncio
    async def test_creates_standard_wrapper_when_blocking_disabled(self, mock_state, mock_generate):
        """Test that TranscriptSyncingModelWrapper is created when blocking_config.enabled=False."""
        metadata = {
            MetadataKeys.SESSION_ID: "session-123",
            MetadataKeys.EPISODE_ID: "episode-456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.SAMPLE_ID: "sample-1",
            "blocking_config": {
                "enabled": False,
            }
        }

        mock_state.metadata = metadata
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_model = MagicMock()
                mock_active_model.return_value = mock_model

                with patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_standard_wrapper:
                    mock_standard_wrapper.return_value = MagicMock()

                    with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                        solver = create_saber_solver("test-agent", mock_factory)
                        await solver(mock_state, mock_generate)

                        # Verify TranscriptSyncingModelWrapper was created
                        mock_standard_wrapper.assert_called_once()
                        call_args = mock_standard_wrapper.call_args
                        assert call_args[1]["base_model"] == mock_model
                        assert call_args[1]["session_id"] == "session-123"
                        assert call_args[1]["episode_id"] == "episode-456"
                        assert call_args[1]["rest_url"] == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_creates_standard_wrapper_when_no_blocking_config(self, mock_state, mock_generate):
        """Test that TranscriptSyncingModelWrapper is created when blocking_config is absent."""
        metadata = {
            MetadataKeys.SESSION_ID: "session-123",
            MetadataKeys.EPISODE_ID: "episode-456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.SAMPLE_ID: "sample-1",
            # No blocking_config
        }

        mock_state.metadata = metadata
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_model = MagicMock()
                mock_active_model.return_value = mock_model

                with patch("saber.inspect_ai.agents.solver_factory.TranscriptSyncingModelWrapper") as mock_standard_wrapper:
                    mock_standard_wrapper.return_value = MagicMock()

                    with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                        solver = create_saber_solver("test-agent", mock_factory)
                        await solver(mock_state, mock_generate)

                        # Verify TranscriptSyncingModelWrapper was created (not blocking)
                        mock_standard_wrapper.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocking_wrapper_uses_default_values(self, mock_state, mock_generate):
        """Test that blocking wrapper uses default values when not specified."""
        metadata = {
            MetadataKeys.SESSION_ID: "session-123",
            MetadataKeys.EPISODE_ID: "episode-456",
            MetadataKeys.SABER_DOMAIN_SLUG: "test-domain",
            MetadataKeys.INSTRUCTION_PROMPT: "Test instruction",
            MetadataKeys.ASSISTANT_PROMPT: "Test assistant",
            MetadataKeys.SUBMIT_PROMPT: "Test submit",
            MetadataKeys.SAMPLE_ID: "sample-1",
            "blocking_config": {
                "enabled": True,
                # No other fields specified - should use defaults
            }
        }

        mock_state.metadata = metadata
        mock_state.store.get.side_effect = lambda key, default=None: {
            InspectStoreKeys.DOMAIN_SLUG: "test-domain",
            InspectStoreKeys.SESSION_ID: "session-123",
            InspectStoreKeys.EPISODE_MAPPING: {
                "sample-1": MagicMock(episode_id="episode-456")
            },
        }.get(key, default)

        mock_agent = AsyncMock(return_value=mock_state)
        mock_factory = MagicMock(return_value=lambda **kwargs: mock_agent)

        with patch("saber.inspect_ai.agents.solver_factory.get_active_domain") as mock_get_domain:
            mock_get_domain.return_value = {"rest_url": "http://localhost:8000"}

            with patch("saber.inspect_ai.agents.solver_factory.active_model") as mock_active_model:
                mock_model = MagicMock()
                mock_active_model.return_value = mock_model

                with patch("saber.inspect_ai.agents.solver_factory.BlockingTranscriptSyncingModelWrapper") as mock_blocking_wrapper:
                    mock_blocking_wrapper.return_value = MagicMock()

                    with patch("saber.inspect_ai.agents.solver_factory.active_model_context_var"):
                        solver = create_saber_solver("test-agent", mock_factory)
                        await solver(mock_state, mock_generate)

                        # Verify defaults were used
                        call_args = mock_blocking_wrapper.call_args
                        assert call_args[1]["poll_interval"] == 2.0  # default
                        assert call_args[1]["max_iterations"] == 50  # default
                        assert call_args[1]["timeout"] == 100.0  # default
                        assert call_args[1]["skip_first_iteration"] is True  # default
