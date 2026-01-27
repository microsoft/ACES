"""Agent directory uploader for SABER evaluations.

This module provides functionality to upload an agent's directory (containing
agent.md, skills, etc.) to the episode container's .github folder, making
the agent configuration available to the agent running inside the sandbox.
"""

import base64
import io
import os
import tarfile
from pathlib import Path

import httpx

from saber.logging_config import LogCategory, get_saber_logger

logger = get_saber_logger(LogCategory.AGENT, __name__)


def create_tar_from_directory(source_dir: Path, archive_name: str = ".github") -> bytes:
    """Create a tar archive from a directory.

    Args:
        source_dir: Path to the directory to archive.
        archive_name: Name of the root directory in the archive.

    Returns:
        Raw bytes of the tar archive.

    Raises:
        FileNotFoundError: If source_dir does not exist.
        ValueError: If source_dir is not a directory.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        for root, dirs, files in os.walk(source_dir, followlinks=False):
            current_dir = Path(root)

            # Skip symlinks
            if current_dir.is_symlink():
                logger.warning(f"Skipping symlink directory: {current_dir}")
                continue

            # Filter out symlink directories
            dirs[:] = [d for d in dirs if not (current_dir / d).is_symlink()]

            for file_name in files:
                file_path = current_dir / file_name

                # Skip symlinks
                if file_path.is_symlink():
                    logger.warning(f"Skipping symlink file: {file_path}")
                    continue

                # Calculate archive path
                rel_path = file_path.relative_to(source_dir)
                arcname = f"{archive_name}/{rel_path}"

                # Read and add file
                with open(file_path, "rb") as f:
                    data = f.read()

                tarinfo = tarfile.TarInfo(name=arcname)
                tarinfo.size = len(data)
                tarinfo.mtime = int(file_path.stat().st_mtime)

                # Set executable mode for scripts
                if file_path.suffix in {".sh", ".py", ".bash"}:
                    tarinfo.mode = 0o755
                else:
                    tarinfo.mode = 0o644

                tar.addfile(tarinfo, io.BytesIO(data))

    tar_buffer.seek(0)
    return tar_buffer.read()


async def upload_agent_directory_to_episode(
    agent_dir: Path,
    session_id: str,
    episode_id: str,
    rest_url: str,
    destination_path: str = "/workspace",
    archive_name: str = "copilot",
    timeout: float = 30.0,
) -> bool:
    """Upload an agent directory to the episode container.

    Creates a tar archive of the agent directory and uploads it to the
    episode container via the SABER REST API.

    Args:
        agent_dir: Path to the agent directory to upload.
        session_id: SABER session ID.
        episode_id: SABER episode ID.
        rest_url: Base URL of the SABER REST API.
        destination_path: Destination path in container (default: /workspace).
        archive_name: Name of the directory in the container (default: copilot).
            Using "copilot" instead of ".github" so files are visible with 'ls'.
        timeout: Request timeout in seconds.

    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        logger.info(
            "Uploading agent directory to episode",
            extra={
                "agent_dir": str(agent_dir),
                "session_id": session_id,
                "episode_id": episode_id,
                "destination_path": destination_path,
                "archive_name": archive_name,
            },
        )

        # Create tar archive
        tar_bytes = create_tar_from_directory(agent_dir, archive_name)
        tar_data_b64 = base64.b64encode(tar_bytes).decode("utf-8")

        logger.debug(
            "Created tar archive",
            extra={
                "tar_size_bytes": len(tar_bytes),
                "b64_size_chars": len(tar_data_b64),
            },
        )

        # Upload via REST API
        url = f"{rest_url}/api/v1/session/{session_id}/episodes/{episode_id}/files"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={
                    "tar_data": tar_data_b64,
                    "destination_path": destination_path,
                },
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    "Agent directory uploaded successfully",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "bytes_copied": result.get("bytes_copied", 0),
                        "destination_path": result.get("destination_path"),
                    },
                )
                return True
            else:
                logger.error(
                    "Failed to upload agent directory",
                    extra={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "status_code": response.status_code,
                        "detail": response.text,
                    },
                )
                return False

    except FileNotFoundError as e:
        logger.error(
            "Agent directory not found",
            extra={"agent_dir": str(agent_dir), "error": str(e)},
        )
        return False
    except httpx.RequestError as e:
        logger.error(
            "HTTP request failed while uploading agent directory",
            extra={
                "session_id": session_id,
                "episode_id": episode_id,
                "error": str(e),
            },
        )
        return False
    except Exception as e:
        logger.error(
            "Unexpected error uploading agent directory",
            extra={
                "session_id": session_id,
                "episode_id": episode_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        return False


def derive_agent_dir_from_skills_dir(skills_dir: str) -> Path | None:
    """Derive the agent directory from a skills directory path.

    The agent directory is assumed to be the parent of the skills directory.

    Args:
        skills_dir: Path to the skills directory.

    Returns:
        Path to the agent directory, or None if it cannot be determined.
    """
    skills_path = Path(skills_dir).resolve()

    if not skills_path.exists():
        logger.warning(f"Skills directory does not exist: {skills_path}")
        return None

    if not skills_path.is_dir():
        logger.warning(f"Skills path is not a directory: {skills_path}")
        return None

    # Agent directory is the parent of skills directory
    agent_dir = skills_path.parent

    # Verify it looks like an agent directory (has agent.md)
    if not (agent_dir / "agent.md").exists():
        logger.warning(
            f"Agent directory does not contain agent.md: {agent_dir}. "
            "Will still upload but this may not be the intended directory."
        )

    return agent_dir
