"""Tests for agent directory uploader.

Tests the functionality to upload agent directories to episode containers.
"""

import base64
import io
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from saber.inspect_ai.agents.agent_directory_uploader import (
    create_tar_from_directory,
    derive_agent_dir_from_skills_dir,
    upload_agent_directory_to_episode,
)


class TestCreateTarFromDirectory:
    """Test create_tar_from_directory function."""

    def test_creates_valid_tar_archive(self, tmp_path):
        """Test that a valid tar archive is created from a directory."""
        # Create test directory structure
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent\n\nThis is a test agent.")
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "skill1.md").write_text("# Skill 1\n\nDescription.")

        # Create tar
        tar_bytes = create_tar_from_directory(agent_dir, archive_name=".github")

        # Verify it's a valid tar
        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            names = tar.getnames()
            assert ".github/agent.md" in names
            assert ".github/skills/skill1.md" in names

    def test_preserves_file_content(self, tmp_path):
        """Test that file content is preserved in the archive."""
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        content = "Hello, world! This is test content."
        (agent_dir / "test.txt").write_text(content)

        tar_bytes = create_tar_from_directory(agent_dir, archive_name=".github")

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            member = tar.getmember(".github/test.txt")
            extracted = tar.extractfile(member)
            assert extracted.read().decode("utf-8") == content

    def test_sets_executable_mode_for_scripts(self, tmp_path):
        """Test that .sh, .py, .bash files get executable permissions."""
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "script.sh").write_text("#!/bin/bash\necho hello")
        (agent_dir / "script.py").write_text("#!/usr/bin/env python\nprint('hello')")
        (agent_dir / "data.txt").write_text("just data")

        tar_bytes = create_tar_from_directory(agent_dir, archive_name=".github")

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            sh_member = tar.getmember(".github/script.sh")
            py_member = tar.getmember(".github/script.py")
            txt_member = tar.getmember(".github/data.txt")

            assert sh_member.mode == 0o755
            assert py_member.mode == 0o755
            assert txt_member.mode == 0o644

    def test_raises_for_nonexistent_directory(self):
        """Test that FileNotFoundError is raised for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            create_tar_from_directory(Path("/nonexistent/path"))

    def test_raises_for_file_not_directory(self, tmp_path):
        """Test that ValueError is raised when path is a file, not directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            create_tar_from_directory(file_path)

    def test_custom_archive_name(self, tmp_path):
        """Test that custom archive name is used."""
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "file.txt").write_text("content")

        tar_bytes = create_tar_from_directory(agent_dir, archive_name="custom-name")

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            names = tar.getnames()
            assert "custom-name/file.txt" in names


class TestDeriveAgentDirFromSkillsDir:
    """Test derive_agent_dir_from_skills_dir function."""

    def test_returns_parent_of_skills_dir(self, tmp_path):
        """Test that parent directory is returned."""
        agent_dir = tmp_path / "test-agent"
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir(parents=True)
        (agent_dir / "agent.md").write_text("# Agent")

        result = derive_agent_dir_from_skills_dir(str(skills_dir))

        assert result == agent_dir

    def test_returns_none_for_nonexistent_path(self, tmp_path):
        """Test that None is returned for nonexistent path."""
        result = derive_agent_dir_from_skills_dir(str(tmp_path / "nonexistent" / "skills"))
        assert result is None

    def test_returns_none_for_file_path(self, tmp_path):
        """Test that None is returned when path is a file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        result = derive_agent_dir_from_skills_dir(str(file_path))
        assert result is None

    def test_warns_when_no_agent_md(self, tmp_path, caplog):
        """Test that warning is logged when agent.md is missing."""
        skills_dir = tmp_path / "test-agent" / "skills"
        skills_dir.mkdir(parents=True)
        # Note: no agent.md created

        result = derive_agent_dir_from_skills_dir(str(skills_dir))

        # Should still return the directory
        assert result == tmp_path / "test-agent"


class TestUploadAgentDirectoryToEpisode:
    """Test upload_agent_directory_to_episode function."""

    @pytest.mark.asyncio
    async def test_successful_upload(self, tmp_path):
        """Test successful upload returns True."""
        import httpx

        # Create test agent directory
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent")

        # Create mock response
        mock_response = httpx.Response(
            status_code=200,
            json={
                "message": "File uploaded successfully",
                "bytes_copied": 1024,
                "destination_path": "/workspace",
            },
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await upload_agent_directory_to_episode(
                agent_dir=agent_dir,
                session_id="session-123",
                episode_id="episode-456",
                rest_url="http://localhost:8000",
            )

            assert result is True
            mock_post.assert_called_once()

            # Verify the request payload
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs
            assert "tar_data" in call_kwargs["json"]
            assert call_kwargs["json"]["destination_path"] == "/workspace"

    @pytest.mark.asyncio
    async def test_failed_upload_returns_false(self, tmp_path):
        """Test that failed upload returns False."""
        import httpx

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent")

        mock_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await upload_agent_directory_to_episode(
                agent_dir=agent_dir,
                session_id="session-123",
                episode_id="episode-456",
                rest_url="http://localhost:8000",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_nonexistent_directory_returns_false(self, tmp_path):
        """Test that nonexistent directory returns False."""
        result = await upload_agent_directory_to_episode(
            agent_dir=tmp_path / "nonexistent",
            session_id="session-123",
            episode_id="episode-456",
            rest_url="http://localhost:8000",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_http_error_returns_false(self, tmp_path):
        """Test that HTTP errors return False."""
        import httpx

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent")

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.RequestError("Connection failed")

            result = await upload_agent_directory_to_episode(
                agent_dir=agent_dir,
                session_id="session-123",
                episode_id="episode-456",
                rest_url="http://localhost:8000",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_correct_url_construction(self, tmp_path):
        """Test that the correct URL is constructed."""
        import httpx

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent")

        mock_response = httpx.Response(
            status_code=200,
            json={"bytes_copied": 100},
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await upload_agent_directory_to_episode(
                agent_dir=agent_dir,
                session_id="my-session",
                episode_id="my-episode",
                rest_url="http://saber.local:9000",
            )

            call_args = mock_post.call_args
            url = call_args.args[0]
            assert url == "http://saber.local:9000/api/v1/session/my-session/episodes/my-episode/files"

    @pytest.mark.asyncio
    async def test_tar_data_is_base64_encoded(self, tmp_path):
        """Test that tar data is properly base64 encoded."""
        import httpx

        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# Test Agent")

        mock_response = httpx.Response(
            status_code=200,
            json={"bytes_copied": 100},
        )

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await upload_agent_directory_to_episode(
                agent_dir=agent_dir,
                session_id="session",
                episode_id="episode",
                rest_url="http://localhost:8000",
            )

            call_kwargs = mock_post.call_args.kwargs
            tar_data_b64 = call_kwargs["json"]["tar_data"]

            # Verify it's valid base64
            decoded = base64.b64decode(tar_data_b64)

            # Verify it's a valid tar
            tar_buffer = io.BytesIO(decoded)
            with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
                names = tar.getnames()
                # Default archive_name is "copilot" (visible with ls, unlike .github)
                assert "copilot/agent.md" in names
