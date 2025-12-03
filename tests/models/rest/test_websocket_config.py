"""Tests for WebSocketConfig dataclass.

Tests cover:
- Configuration defaults
- Validation (positive values, max > initial)
- from_blocking_config() mapping
- Backward compatibility
- Push/Pull configuration separation
"""

import pytest

from saber.models.rest.websocket_config import WebSocketConfig, PushConfig, PullConfig


class TestWebSocketConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Test that defaults match expected values."""
        config = WebSocketConfig()

        # Connection timeouts
        assert config.connection_timeout == 10.0

        # Keepalive
        assert config.ping_interval == 30.0
        assert config.pong_timeout == 10.0

        # Reconnection
        assert config.reconnect_enabled is True
        assert config.max_reconnect_attempts == 3
        assert config.initial_reconnect_delay == 1.0
        assert config.max_reconnect_delay == 30.0
        assert config.reconnect_backoff_multiplier == 2.0

        # Push configuration
        assert config.push.enabled is True
        assert config.push.confirmation_timeout == 5.0
        assert config.push.retry_enabled is True
        assert config.push.max_retry_attempts == 3
        assert config.push.retry_backoff_multiplier == 2.0

        # Pull configuration
        assert config.pull.enabled is True
        assert config.pull.event_timeout == 300.0
        assert config.pull.sync_timeout == 5.0
        assert config.pull.event_queue_max_size == 100

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = WebSocketConfig(
            connection_timeout=5.0,
            max_reconnect_attempts=5,
            push=PushConfig(confirmation_timeout=3.0, max_retry_attempts=5),
            pull=PullConfig(event_timeout=60.0, event_queue_max_size=200),
        )

        assert config.connection_timeout == 5.0
        assert config.max_reconnect_attempts == 5
        assert config.push.confirmation_timeout == 3.0
        assert config.push.max_retry_attempts == 5
        assert config.pull.event_timeout == 60.0
        assert config.pull.event_queue_max_size == 200


class TestWebSocketConfigValidation:
    """Test configuration validation."""

    def test_negative_connection_timeout_raises(self):
        """Test that negative connection timeout raises ValueError."""
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            WebSocketConfig(connection_timeout=-1.0)

    def test_negative_reconnect_delay_raises(self):
        """Test that negative initial_reconnect_delay raises ValueError."""
        with pytest.raises(ValueError, match="initial_reconnect_delay must be positive"):
            WebSocketConfig(initial_reconnect_delay=-1.0)

    def test_negative_max_reconnect_delay_raises(self):
        """Test that negative max_reconnect_delay raises ValueError."""
        with pytest.raises(ValueError, match="max_reconnect_delay must be positive"):
            WebSocketConfig(max_reconnect_delay=-1.0)

    def test_max_delay_less_than_initial_raises(self):
        """Test that max_reconnect_delay < initial_reconnect_delay raises ValueError."""
        with pytest.raises(ValueError, match="max_reconnect_delay.*must be >= initial_reconnect_delay"):
            WebSocketConfig(
                initial_reconnect_delay=10.0,
                max_reconnect_delay=5.0,
            )

    def test_negative_backoff_multiplier_raises(self):
        """Test that negative backoff multiplier raises ValueError."""
        with pytest.raises(ValueError, match="reconnect_backoff_multiplier must be positive"):
            WebSocketConfig(reconnect_backoff_multiplier=-1.0)

    def test_zero_max_attempts_raises(self):
        """Test that zero max_reconnect_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_reconnect_attempts must be at least 1"):
            WebSocketConfig(max_reconnect_attempts=0)

    def test_zero_ping_interval_disables_keepalive(self):
        """Test that ping_interval=None is allowed (disables keepalive)."""
        config = WebSocketConfig(ping_interval=None)
        assert config.ping_interval is None

    def test_negative_pong_timeout_raises(self):
        """Test that negative pong_timeout raises ValueError."""
        with pytest.raises(ValueError, match="pong_timeout must be positive"):
            WebSocketConfig(pong_timeout=-1.0)


class TestPushConfigValidation:
    """Test push configuration validation."""

    def test_negative_confirmation_timeout_raises(self):
        """Test that negative confirmation timeout raises ValueError."""
        with pytest.raises(ValueError, match="confirmation_timeout must be positive"):
            PushConfig(confirmation_timeout=-1.0)

    def test_negative_max_retry_attempts_raises(self):
        """Test that negative max_retry_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_retry_attempts must be non-negative"):
            PushConfig(max_retry_attempts=-1)

    def test_negative_retry_backoff_multiplier_raises(self):
        """Test that negative retry_backoff_multiplier raises ValueError."""
        with pytest.raises(ValueError, match="retry_backoff_multiplier must be positive"):
            PushConfig(retry_backoff_multiplier=-1.0)


class TestPullConfigValidation:
    """Test pull configuration validation."""

    def test_negative_event_timeout_raises(self):
        """Test that negative event timeout raises ValueError."""
        with pytest.raises(ValueError, match="event_timeout must be positive"):
            PullConfig(event_timeout=-1.0)

    def test_negative_sync_timeout_raises(self):
        """Test that negative sync timeout raises ValueError."""
        with pytest.raises(ValueError, match="sync_timeout must be positive"):
            PullConfig(sync_timeout=-1.0)

    def test_negative_queue_size_raises(self):
        """Test that negative event_queue_max_size raises ValueError."""
        with pytest.raises(ValueError, match="event_queue_max_size must be positive"):
            PullConfig(event_queue_max_size=-1)


class TestWebSocketConfigFromBlockingConfig:
    """Test from_blocking_config() factory method."""

    def test_from_blocking_config_empty_dict(self):
        """Test that empty dict uses all defaults."""
        config = WebSocketConfig.from_blocking_config({})

        # Should match defaults
        assert config.connection_timeout == 10.0
        assert config.reconnect_enabled is True
        assert config.push.confirmation_timeout == 5.0
        assert config.pull.event_timeout == 300.0

    def test_from_blocking_config_with_ws_prefixed_keys(self):
        """Test mapping ws_* keys from domain.yaml."""
        blocking_config = {
            "ws_connection_timeout": 15.0,
            "ws_reconnect_enabled": False,
            "ws_max_reconnect_attempts": 5,
            "ws_initial_reconnect_delay": 2.0,
            "ws_max_reconnect_delay": 60.0,
            "ws_reconnect_backoff_multiplier": 1.5,
            "ws_ping_interval": 20.0,
            "ws_pong_timeout": 15.0,
            # Push config
            "ws_push_enabled": True,
            "ws_confirmation_timeout": 3.0,
            "ws_push_retry_enabled": True,
            "ws_push_max_retry_attempts": 5,
            "ws_push_retry_backoff_multiplier": 1.5,
            # Pull config
            "ws_pull_enabled": True,
            "ws_event_timeout": 60.0,
            "ws_sync_timeout": 10.0,
            "ws_event_queue_max_size": 50,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        # Connection config
        assert config.connection_timeout == 15.0
        assert config.reconnect_enabled is False
        assert config.max_reconnect_attempts == 5
        assert config.initial_reconnect_delay == 2.0
        assert config.max_reconnect_delay == 60.0
        assert config.reconnect_backoff_multiplier == 1.5
        assert config.ping_interval == 20.0
        assert config.pong_timeout == 15.0

        # Push config
        assert config.push.enabled is True
        assert config.push.confirmation_timeout == 3.0
        assert config.push.retry_enabled is True
        assert config.push.max_retry_attempts == 5
        assert config.push.retry_backoff_multiplier == 1.5

        # Pull config
        assert config.pull.enabled is True
        assert config.pull.event_timeout == 60.0
        assert config.pull.sync_timeout == 10.0
        assert config.pull.event_queue_max_size == 50

    def test_from_blocking_config_partial_override(self):
        """Test that only specified keys override defaults."""
        blocking_config = {
            "ws_connection_timeout": 20.0,
            "ws_max_reconnect_attempts": 10,
            "ws_event_timeout": 120.0,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        # Overridden values
        assert config.connection_timeout == 20.0
        assert config.max_reconnect_attempts == 10
        assert config.pull.event_timeout == 120.0

        # Default values
        assert config.push.confirmation_timeout == 5.0  # Default
        assert config.pull.sync_timeout == 5.0          # Default
        assert config.reconnect_enabled is True          # Default

    def test_from_blocking_config_ignores_non_ws_keys(self):
        """Test that non-ws_* keys are ignored."""
        blocking_config = {
            "enabled": True,
            "use_websocket": True,
            "skip_first_iteration": False,
            "ws_connection_timeout": 25.0,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        # Only ws_* key should be mapped
        assert config.connection_timeout == 25.0

        # Defaults for unmapped keys
        assert config.confirmation_timeout == 5.0

    def test_from_blocking_config_validates(self):
        """Test that from_blocking_config() validates values."""
        blocking_config = {
            "ws_connection_timeout": -5.0,  # Invalid
        }

        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            WebSocketConfig.from_blocking_config(blocking_config)

    def test_from_blocking_config_none_value_keeps_default(self):
        """Test that None values keep defaults."""
        blocking_config = {
            "ws_connection_timeout": None,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        # Should use default, not None
        assert config.connection_timeout == 10.0


class TestWebSocketConfigBackwardCompatibility:
    """Test backward compatibility scenarios."""

    def test_legacy_domain_yaml_works(self):
        """Test that domain.yaml without ws_* keys works (uses defaults)."""
        # Legacy domain.yaml only has:
        legacy_blocking_config = {
            "enabled": True,
            "use_websocket": True,
            "skip_first_iteration": True,
        }

        # Should not raise
        config = WebSocketConfig.from_blocking_config(legacy_blocking_config)

        # All defaults
        assert config.connection_timeout == 10.0
        assert config.reconnect_enabled is True

    def test_disable_reconnection_legacy(self):
        """Test disabling reconnection for legacy behavior."""
        blocking_config = {
            "ws_reconnect_enabled": False,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        assert config.reconnect_enabled is False

    def test_set_max_attempts_to_one_legacy(self):
        """Test setting max_reconnect_attempts=1 for legacy behavior (no retries)."""
        blocking_config = {
            "ws_max_reconnect_attempts": 1,
        }

        config = WebSocketConfig.from_blocking_config(blocking_config)

        assert config.max_reconnect_attempts == 1


class TestWebSocketConfigReconnectionBehavior:
    """Test reconnection configuration behavior."""

    def test_exponential_backoff_calculation(self):
        """Test that backoff config supports exponential backoff calculation."""
        config = WebSocketConfig(
            initial_reconnect_delay=1.0,
            reconnect_backoff_multiplier=2.0,
            max_reconnect_delay=30.0,
        )

        # Simulate backoff calculation (as done in model_wrapper.py)
        delays = []
        for attempt in range(5):
            delay = min(
                config.initial_reconnect_delay * (config.reconnect_backoff_multiplier ** attempt),
                config.max_reconnect_delay
            )
            delays.append(delay)

        # Should be: 1.0, 2.0, 4.0, 8.0, 16.0
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0
        assert delays[3] == 8.0
        assert delays[4] == 16.0

    def test_backoff_capped_by_max_delay(self):
        """Test that backoff is capped by max_reconnect_delay."""
        config = WebSocketConfig(
            initial_reconnect_delay=10.0,
            reconnect_backoff_multiplier=2.0,
            max_reconnect_delay=15.0,
        )

        # Attempt 2: 10 * 2^1 = 20, capped at 15
        delay = min(
            config.initial_reconnect_delay * (config.reconnect_backoff_multiplier ** 1),
            config.max_reconnect_delay
        )

        assert delay == 15.0  # Capped


class TestWebSocketConfigQueueManagement:
    """Test event queue configuration."""

    def test_queue_size_configurable(self):
        """Test that queue size can be configured."""
        small_queue = WebSocketConfig(pull=PullConfig(event_queue_max_size=10))
        large_queue = WebSocketConfig(pull=PullConfig(event_queue_max_size=1000))

        assert small_queue.pull.event_queue_max_size == 10
        assert large_queue.pull.event_queue_max_size == 1000

    def test_queue_size_default_reasonable(self):
        """Test that default queue size is reasonable (100)."""
        config = WebSocketConfig()
        assert config.pull.event_queue_max_size == 100


class TestWebSocketConfigKeepalive:
    """Test keepalive configuration."""

    def test_keepalive_enabled_by_default(self):
        """Test that keepalive is enabled by default (ping_interval=30s)."""
        config = WebSocketConfig()
        assert config.ping_interval == 30.0
        assert config.pong_timeout == 10.0

    def test_keepalive_can_be_disabled(self):
        """Test that keepalive can be disabled (ping_interval=None)."""
        config = WebSocketConfig(ping_interval=None)
        assert config.ping_interval is None

    def test_custom_keepalive_intervals(self):
        """Test custom keepalive intervals."""
        config = WebSocketConfig(
            ping_interval=60.0,
            pong_timeout=20.0,
        )

        assert config.ping_interval == 60.0
        assert config.pong_timeout == 20.0
