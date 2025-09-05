import asyncio
import json
from typing import List

import pytest

from saber.client.api.rest_client import SABERRestClient


@pytest.mark.asyncio
async def test_progress_stream_start_stop(monkeypatch):
    # Use a fake sidecar url; we won't actually connect, just ensure start/stop wiring works
    client = SABERRestClient(base_url="http://localhost:8000", sidecar_url="http://localhost:8001")

    # Set a session ID to pass the validation
    client.session_id = "test-session-123"

    received: List[dict] = []

    def cb(evt: dict) -> None:
        received.append(evt)

    # Monkeypatch sync runner used in the thread to a short no-op
    def fake_run_sync(agent_id: str, loop):
        pass

    monkeypatch.setattr(client, "_run_progress_stream_sync", fake_run_sync)

    ok = await client.start_progress_stream(cb, episode_id="test-episode-123", agent_id="test-agent")
    assert ok is True

    await client.stop_progress_stream()
    # Thread should be cleared
    assert getattr(client, "_sse_thread", None) is None
