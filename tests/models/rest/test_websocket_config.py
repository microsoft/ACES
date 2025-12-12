"""Tests for WebSocketConfig dataclass.

Tests cover:
- Configuration defaults
- Validation (positive values, max > initial)
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

        # Push configuration - only enabled is configurable
        assert config.push.enabled is True
        assert config.push.confirmation_timeout == 5.0  # Internal default
        assert config.push.retry_backoff_multiplier == 2.0  # Internal default

        # Pull configuration - only enabled is configurable
        assert config.pull.enabled is True
        assert config.pull.event_timeout == 300.0  # Internal default
        assert config.pull.sync_timeout == 5.0  # Internal default
        assert config.pull.event_queue_max_size == 100  # Internal default

    def test_custom_enabled_values(self):
        """Test creating config with custom enabled values."""
        config = WebSocketConfig(
            connection_timeout=5.0,
            max_reconnect_attempts=5,
            push=PushConfig(enabled=False),
            pull=PullConfig(enabled=False),
        )

        assert config.connection_timeout == 5.0
        assert config.max_reconnect_attempts == 5
        assert config.push.enabled is False
        assert config.pull.enabled is False


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


class TestPushConfigDefaults:
    """Test push configuration defaults."""

    def test_push_enabled_by_default(self):
        """Test that push is enabled by default."""
        config = PushConfig()
        assert config.enabled is True

    def test_push_can_be_disabled(self):
        """Test that push can be disabled."""
        config = PushConfig(enabled=False)
        assert config.enabled is False


class TestPullConfigDefaults:
    """Test pull configuration defaults."""

    def test_pull_enabled_by_default(self):
        """Test that pull is enabled by default."""
        config = PullConfig()
        assert config.enabled is True

    def test_pull_can_be_disabled(self):
        """Test that pull can be disabled."""
        config = PullConfig(enabled=False)
        assert config.enabled is False


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

    def test_queue_size_has_reasonable_default(self):
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
