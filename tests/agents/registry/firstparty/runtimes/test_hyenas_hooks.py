"""Tests for Hyenas sandbox hook functions (repo injection & findings extraction)."""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_tar_gz_bytes(file_name: str = "hello.txt", content: bytes = b"hello") -> bytes:
    """Create a minimal .tar.gz archive in memory and return raw bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=file_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


@pytest.fixture
def mock_sandbox() -> AsyncMock:
    """Provide an AsyncMock matching the SandboxEnvironment interface."""
    sandbox = AsyncMock()
    sandbox.write_file = AsyncMock()
    sandbox.read_file = AsyncMock()
    sandbox.exec = AsyncMock()
    return sandbox


# ── inject_target_repo ────────────────────────────────────────────────


class TestInjectTargetRepo:
    """Tests for inject_target_repo()."""

    @pytest.mark.asyncio
    async def test_inject_from_tarball(self, mock_sandbox: AsyncMock) -> None:
        """Base64-encoded tarball is decoded, written, and extracted."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        raw = _make_tar_gz_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        exec_result = MagicMock()
        exec_result.returncode = 0
        exec_result.stdout = ""
        exec_result.stderr = ""
        mock_sandbox.exec.return_value = exec_result

        await inject_target_repo(mock_sandbox, {"repo_tarball": b64})

        # Should write decoded bytes to sandbox
        mock_sandbox.write_file.assert_awaited_once_with(
            "/tmp/repo.tar.gz", raw
        )
        # Should call tar to extract, then check if git repo
        calls = mock_sandbox.exec.await_args_list
        assert calls[0].args[0] == ["tar", "xzf", "/tmp/repo.tar.gz", "-C", "/workspace"]
        assert calls[1].args[0] == ["git", "rev-parse", "--git-dir"]

    @pytest.mark.asyncio
    async def test_inject_from_path(
        self, mock_sandbox: AsyncMock, tmp_path: Path
    ) -> None:
        """Host-side tarball path is read, written to sandbox, and extracted."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        raw = _make_tar_gz_bytes()
        tar_file = tmp_path / "repo.tar.gz"
        tar_file.write_bytes(raw)

        exec_result = MagicMock()
        exec_result.returncode = 0
        exec_result.stdout = ""
        exec_result.stderr = ""
        mock_sandbox.exec.return_value = exec_result

        await inject_target_repo(mock_sandbox, {"repo_path": str(tar_file)})

        mock_sandbox.write_file.assert_awaited_once_with(
            "/tmp/repo.tar.gz", raw
        )
        calls = mock_sandbox.exec.await_args_list
        assert calls[0].args[0] == ["tar", "xzf", "/tmp/repo.tar.gz", "-C", "/workspace"]
        assert calls[1].args[0] == ["git", "rev-parse", "--git-dir"]

    @pytest.mark.asyncio
    async def test_inject_tarball_priority(
        self, mock_sandbox: AsyncMock, tmp_path: Path
    ) -> None:
        """When both repo_tarball and repo_path are present, tarball wins."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        tarball_bytes = _make_tar_gz_bytes("from_tarball.txt", b"tarball")
        path_bytes = _make_tar_gz_bytes("from_path.txt", b"path")

        tar_file = tmp_path / "repo.tar.gz"
        tar_file.write_bytes(path_bytes)

        b64 = base64.b64encode(tarball_bytes).decode("ascii")
        exec_result = MagicMock()
        exec_result.returncode = 0
        exec_result.stdout = ""
        exec_result.stderr = ""
        mock_sandbox.exec.return_value = exec_result

        await inject_target_repo(
            mock_sandbox,
            {"repo_tarball": b64, "repo_path": str(tar_file)},
        )

        # Should use tarball bytes, NOT path bytes
        mock_sandbox.write_file.assert_awaited_once_with(
            "/tmp/repo.tar.gz", tarball_bytes
        )

    @pytest.mark.asyncio
    async def test_inject_missing_metadata_raises(
        self, mock_sandbox: AsyncMock
    ) -> None:
        """Empty metadata raises ValueError."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        with pytest.raises(ValueError, match="repo_tarball.*repo_path"):
            await inject_target_repo(mock_sandbox, {})

    @pytest.mark.asyncio
    async def test_inject_tar_failure_raises(
        self, mock_sandbox: AsyncMock
    ) -> None:
        """Non-zero tar exit code raises RuntimeError with stderr."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        raw = _make_tar_gz_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        exec_result = MagicMock()
        exec_result.returncode = 1
        exec_result.stdout = ""
        exec_result.stderr = "tar: corrupted archive"
        mock_sandbox.exec.return_value = exec_result

        with pytest.raises(RuntimeError, match="tar: corrupted archive"):
            await inject_target_repo(mock_sandbox, {"repo_tarball": b64})

    @pytest.mark.asyncio
    async def test_inject_empty_tarball_writes_empty_bytes(
        self, mock_sandbox: AsyncMock
    ) -> None:
        """An empty repo_tarball string decodes to empty bytes; tar may fail."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            inject_target_repo,
        )

        exec_result = MagicMock()
        exec_result.returncode = 1
        exec_result.stdout = ""
        exec_result.stderr = "tar: not a gzip archive"
        mock_sandbox.exec.return_value = exec_result

        # Empty base64 decodes to b""; tar will fail on that
        with pytest.raises(RuntimeError, match="not a gzip archive"):
            await inject_target_repo(mock_sandbox, {"repo_tarball": ""})


# ── extract_hyenas_findings ───────────────────────────────────────────


class TestExtractHyenasFindings:
    """Tests for extract_hyenas_findings()."""

    @pytest.mark.asyncio
    async def test_extract_valid_findings(self, mock_sandbox: AsyncMock) -> None:
        """Multi-line JSONL is parsed into a list of HyenasFinding."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        lines = [
            json.dumps(
                {
                    "file_path": "src/app.py",
                    "function_name": "handle",
                    "cwe_id": "CWE-79",
                    "line_number": 42,
                    "confidence": "LIKELY",
                    "description": "XSS in template",
                }
            ),
            json.dumps(
                {
                    "file_path": "src/db.py",
                    "function_name": "query",
                    "cwe_id": "CWE-89",
                    "line_number": 10,
                    "confidence": "CONFIRMED",
                    "description": "SQL injection",
                }
            ),
        ]
        mock_sandbox.read_file.return_value = "\n".join(lines)

        findings = await extract_hyenas_findings(mock_sandbox)

        assert len(findings) == 2
        assert findings[0].file_path == "src/app.py"
        assert findings[0].cwe_id == "CWE-79"
        assert findings[1].file_path == "src/db.py"
        assert findings[1].confidence == "CONFIRMED"

    @pytest.mark.asyncio
    async def test_extract_empty_file(self, mock_sandbox: AsyncMock) -> None:
        """Empty file content returns empty list."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        mock_sandbox.read_file.return_value = ""

        findings = await extract_hyenas_findings(mock_sandbox)

        assert findings == []

    @pytest.mark.asyncio
    async def test_extract_missing_file(self, mock_sandbox: AsyncMock) -> None:
        """Missing findings file returns empty list (not an error)."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        mock_sandbox.read_file.side_effect = FileNotFoundError(
            "No such file: /output/.hyenas/scan/findings.jsonl"
        )

        findings = await extract_hyenas_findings(mock_sandbox)

        assert findings == []

    @pytest.mark.asyncio
    async def test_extract_mixed_valid_invalid(
        self, mock_sandbox: AsyncMock
    ) -> None:
        """Valid lines are returned; invalid lines are skipped."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        content = "\n".join(
            [
                json.dumps({"file_path": "a.py", "cwe_id": "CWE-1"}),
                "NOT VALID JSON {{{",
                json.dumps({"file_path": "b.py"}),
            ]
        )
        mock_sandbox.read_file.return_value = content

        findings = await extract_hyenas_findings(mock_sandbox)

        assert len(findings) == 2
        assert findings[0].file_path == "a.py"
        assert findings[1].file_path == "b.py"

    @pytest.mark.asyncio
    async def test_extract_partial_fields(self, mock_sandbox: AsyncMock) -> None:
        """Only the required field (file_path) is present; optionals are None."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        mock_sandbox.read_file.return_value = json.dumps({"file_path": "main.c"})

        findings = await extract_hyenas_findings(mock_sandbox)

        assert len(findings) == 1
        f = findings[0]
        assert f.file_path == "main.c"
        assert f.function_name is None
        assert f.cwe_id is None
        assert f.line_number is None
        assert f.confidence is None
        assert f.description is None

    @pytest.mark.asyncio
    async def test_extract_blank_lines_skipped(
        self, mock_sandbox: AsyncMock
    ) -> None:
        """Blank lines interspersed in JSONL are silently ignored."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            extract_hyenas_findings,
        )

        content = "\n".join(
            [
                "",
                json.dumps({"file_path": "x.py"}),
                "",
                "   ",
                json.dumps({"file_path": "y.py"}),
                "",
            ]
        )
        mock_sandbox.read_file.return_value = content

        findings = await extract_hyenas_findings(mock_sandbox)

        assert len(findings) == 2
        assert findings[0].file_path == "x.py"
        assert findings[1].file_path == "y.py"


# ── consolidate_hyenas_findings ───────────────────────────────────────


class TestConsolidateHyenasFindings:
    """Tests for consolidate_hyenas_findings()."""

    @pytest.mark.asyncio
    async def test_consolidate_valid_findings(self, mock_sandbox: AsyncMock) -> None:
        """Two individual JSON files are consolidated into a JSONL file."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        finding_a = json.dumps({"file_path": "a.py", "cwe_id": "CWE-79"})
        finding_b = json.dumps({"file_path": "b.py", "cwe_id": "CWE-89"})

        # find returns two file paths
        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = "/output/.hyenas/findings/a.json\n/output/.hyenas/findings/b.json\n"
        find_result.stderr = ""

        # mkdir returns success
        mkdir_result = MagicMock()
        mkdir_result.returncode = 0
        mkdir_result.stdout = ""
        mkdir_result.stderr = ""

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.side_effect = [finding_a, finding_b]

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 2
        mock_sandbox.write_file.assert_awaited_once()
        written_path = mock_sandbox.write_file.await_args[0][0]
        written_content = mock_sandbox.write_file.await_args[0][1]
        assert written_path == "/output/.hyenas/scan/findings.jsonl"
        lines = written_content.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["file_path"] == "a.py"
        assert json.loads(lines[1])["file_path"] == "b.py"

    @pytest.mark.asyncio
    async def test_consolidate_empty_dir(self, mock_sandbox: AsyncMock) -> None:
        """No JSON files in findings dir returns 0 and writes nothing."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = ""
        find_result.stderr = ""
        mock_sandbox.exec.return_value = find_result

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 0
        mock_sandbox.write_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consolidate_findings_dir_missing(self, mock_sandbox: AsyncMock) -> None:
        """find command fails (directory missing) returns 0."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        find_result = MagicMock()
        find_result.returncode = 1
        find_result.stdout = ""
        find_result.stderr = "No such file or directory"
        mock_sandbox.exec.return_value = find_result

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 0
        mock_sandbox.write_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consolidate_invalid_json_skipped(self, mock_sandbox: AsyncMock) -> None:
        """Invalid JSON files are skipped; only valid entries are written."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        valid_finding = json.dumps({"file_path": "ok.py", "cwe_id": "CWE-79"})
        invalid_content = "NOT VALID JSON {{{"

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = "/output/.hyenas/findings/ok.json\n/output/.hyenas/findings/bad.json\n"
        find_result.stderr = ""

        mkdir_result = MagicMock()
        mkdir_result.returncode = 0
        mkdir_result.stdout = ""
        mkdir_result.stderr = ""

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.side_effect = [valid_finding, invalid_content]

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 1
        written_content = mock_sandbox.write_file.await_args[0][1]
        lines = written_content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["file_path"] == "ok.py"

    @pytest.mark.asyncio
    async def test_consolidate_creates_scan_dir(self, mock_sandbox: AsyncMock) -> None:
        """mkdir -p is called to create the scan directory."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        finding = json.dumps({"file_path": "x.py"})

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = "/output/.hyenas/findings/x.json\n"
        find_result.stderr = ""

        mkdir_result = MagicMock()
        mkdir_result.returncode = 0
        mkdir_result.stdout = ""
        mkdir_result.stderr = ""

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.return_value = finding

        await consolidate_hyenas_findings(mock_sandbox)

        # Second exec call should be mkdir -p for the scan dir
        calls = mock_sandbox.exec.await_args_list
        assert calls[1].args[0] == ["mkdir", "-p", "/output/.hyenas/scan"]

    @pytest.mark.asyncio
    async def test_consolidate_read_file_exception(self, mock_sandbox: AsyncMock) -> None:
        """read_file raising on one file still consolidates the rest."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        valid_finding = json.dumps({"file_path": "good.py", "cwe_id": "CWE-22"})

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = (
            "/output/.hyenas/findings/good.json\n"
            "/output/.hyenas/findings/gone.json\n"
        )
        find_result.stderr = ""

        mkdir_result = MagicMock()
        mkdir_result.returncode = 0
        mkdir_result.stdout = ""
        mkdir_result.stderr = ""

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.side_effect = [
            valid_finding,
            FileNotFoundError("deleted between find and read"),
        ]

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 1
        written_content = mock_sandbox.write_file.await_args[0][1]
        lines = written_content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["file_path"] == "good.py"

    @pytest.mark.asyncio
    async def test_consolidate_non_dict_json_skipped(self, mock_sandbox: AsyncMock) -> None:
        """JSON that is not a dict (array, string, number) is skipped."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        valid_finding = json.dumps({"file_path": "ok.py"})
        array_json = json.dumps([{"file_path": "sneaky.py"}])
        string_json = json.dumps("just a string")

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = (
            "/output/.hyenas/findings/ok.json\n"
            "/output/.hyenas/findings/array.json\n"
            "/output/.hyenas/findings/string.json\n"
        )
        find_result.stderr = ""

        mkdir_result = MagicMock()
        mkdir_result.returncode = 0
        mkdir_result.stdout = ""
        mkdir_result.stderr = ""

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.side_effect = [valid_finding, array_json, string_json]

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 1
        written_content = mock_sandbox.write_file.await_args[0][1]
        lines = written_content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["file_path"] == "ok.py"

    @pytest.mark.asyncio
    async def test_consolidate_mkdir_failure(self, mock_sandbox: AsyncMock) -> None:
        """mkdir -p failure returns 0 and does not write JSONL."""
        from saber.agents.registry.firstparty.runtimes.hyenas_hooks import (
            consolidate_hyenas_findings,
        )

        finding = json.dumps({"file_path": "x.py"})

        find_result = MagicMock()
        find_result.returncode = 0
        find_result.stdout = "/output/.hyenas/findings/x.json\n"
        find_result.stderr = ""

        mkdir_result = MagicMock()
        mkdir_result.returncode = 1
        mkdir_result.stdout = ""
        mkdir_result.stderr = "Permission denied"

        mock_sandbox.exec.side_effect = [find_result, mkdir_result]
        mock_sandbox.read_file.return_value = finding

        count = await consolidate_hyenas_findings(mock_sandbox)

        assert count == 0
        mock_sandbox.write_file.assert_not_awaited()
