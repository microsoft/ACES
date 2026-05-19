# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for saber.environments.images – Phases 2 & 3."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from saber.environments.images import (
    BASE_DOCKERFILE_NAME,
    BASE_IMAGE_TAG,
    ImageBuildError,
    ImageBuildResult,
    PreflightBuildError,
    PreflightResult,
    RebuildMode,
    _BuildProgressDisplay,
    _create_build_progress,
    _detect_build_phase,
    build_domain_images,
    build_image,
    find_base_dockerfile,
    image_exists,
)

# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_base_image_tag(self) -> None:
        assert BASE_IMAGE_TAG == "saber/sandbox:latest"

    def test_base_dockerfile_name(self) -> None:
        assert BASE_DOCKERFILE_NAME == "Dockerfile.saber_sandbox"


# ── ImageBuildResult ─────────────────────────────────────────────────


class TestImageBuildResult:
    def test_creation(self) -> None:
        result = ImageBuildResult(name="img", tag="img:1", action="built")
        assert result.name == "img"
        assert result.tag == "img:1"
        assert result.action == "built"
        assert result.error is None

    def test_creation_with_error(self) -> None:
        result = ImageBuildResult(name="img", tag="img:1", action="failed", error="boom")
        assert result.error == "boom"

    def test_frozen(self) -> None:
        result = ImageBuildResult(name="img", tag="img:1", action="skipped")
        with pytest.raises(ValidationError):
            result.name = "other"  # type: ignore[misc]


# ── ImageBuildError ──────────────────────────────────────────────────


class TestImageBuildError:
    def test_attributes(self) -> None:
        err = ImageBuildError(tag="x:1", stderr="oops", returncode=2)
        assert err.tag == "x:1"
        assert err.stderr == "oops"
        assert err.returncode == 2

    def test_message_contents(self) -> None:
        err = ImageBuildError(tag="x:1", stderr="oops", returncode=2)
        msg = str(err)
        assert "x:1" in msg
        assert "2" in msg
        assert "oops" in msg


# ── PreflightBuildError ──────────────────────────────────────────────


class TestPreflightBuildError:
    def test_single_failure(self) -> None:
        results = [ImageBuildResult(name="db", tag="saber/db:latest", action="failed", error="COPY failed")]
        err = PreflightBuildError(results)
        msg = str(err)
        assert "PREFLIGHT IMAGE BUILD FAILED" in msg
        assert "1 image(s) failed to build: db" in msg
        assert "COPY failed" in msg
        assert "saber/db:latest" in msg

    def test_multiple_failures(self) -> None:
        results = [
            ImageBuildResult(name="db", tag="db:1", action="failed", error="missing file"),
            ImageBuildResult(name="web", tag="web:1", action="failed", error="apt failed"),
        ]
        err = PreflightBuildError(results)
        msg = str(err)
        assert "2 image(s) failed to build: db, web" in msg
        assert "missing file" in msg
        assert "apt failed" in msg

    def test_is_runtime_error(self) -> None:
        results = [ImageBuildResult(name="x", tag="x:1", action="failed", error="err")]
        err = PreflightBuildError(results)
        assert isinstance(err, RuntimeError)

    def test_long_error_truncated(self) -> None:
        long_error = "\n".join(f"line {i}" for i in range(50))
        results = [ImageBuildResult(name="big", tag="big:1", action="failed", error=long_error)]
        err = PreflightBuildError(results)
        msg = str(err)
        assert "truncated" in msg
        # Last line should be present
        assert "line 49" in msg

    def test_no_error_detail(self) -> None:
        results = [ImageBuildResult(name="x", tag="x:1", action="failed", error=None)]
        err = PreflightBuildError(results)
        msg = str(err)
        assert "no error details captured" in msg


# ── image_exists ─────────────────────────────────────────────────────


class TestImageExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_image_present(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = await image_exists("test:latest")

        assert result is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0]
        assert call_args[:4] == ("docker", "image", "inspect", "test:latest")

    @pytest.mark.asyncio
    async def test_returns_false_when_image_absent(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await image_exists("missing:tag")

        assert result is False


# ── build_image ──────────────────────────────────────────────────────


def _make_stream_proc(
    stdout_lines: list[bytes] | None = None,
    stderr_lines: list[bytes] | None = None,
    returncode: int = 0,
) -> AsyncMock:
    """Create a mock subprocess with streaming stdout/stderr for build_image tests."""
    mock_proc = AsyncMock()

    stdout_reader = AsyncMock()
    stdout_reader.readline = AsyncMock(side_effect=[*(stdout_lines or []), b""])
    mock_proc.stdout = stdout_reader

    stderr_reader = AsyncMock()
    stderr_reader.readline = AsyncMock(side_effect=[*(stderr_lines or []), b""])
    mock_proc.stderr = stderr_reader

    mock_proc.wait = AsyncMock(return_value=returncode)
    mock_proc.returncode = returncode

    return mock_proc


class TestBuildImage:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        mock_proc = _make_stream_proc()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"))

        call_args = mock_exec.call_args[0]
        assert "docker" in call_args
        assert "build" in call_args
        assert "-f" in call_args
        assert "/df" in call_args
        assert "-t" in call_args
        assert "t:1" in call_args
        assert "/ctx" in call_args

    @pytest.mark.asyncio
    async def test_failure_raises(self) -> None:
        mock_proc = _make_stream_proc(
            stderr_lines=[b"error msg\n"],
            returncode=1,
        )

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(ImageBuildError) as exc_info:
                await build_image(tag="fail:1", dockerfile=Path("/df"), context=Path("/ctx"))

        assert exc_info.value.tag == "fail:1"
        assert "error msg" in exc_info.value.stderr
        assert exc_info.value.returncode == 1

    @pytest.mark.asyncio
    async def test_default_context_is_dockerfile_parent(self) -> None:
        mock_proc = _make_stream_proc()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/some/dir/Dockerfile"))

        call_args = mock_exec.call_args[0]
        # Last positional arg should be the context (dockerfile's parent)
        assert call_args[-1] == "/some/dir"

    @pytest.mark.asyncio
    async def test_with_build_args_and_labels(self) -> None:
        mock_proc = _make_stream_proc()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(
                tag="t:1",
                dockerfile=Path("/df"),
                context=Path("/ctx"),
                build_args={"K1": "V1"},
                labels={"L1": "V2"},
            )

        call_args = mock_exec.call_args[0]
        args_list = list(call_args)
        assert "--build-arg" in args_list
        assert "K1=V1" in args_list
        assert "--label" in args_list
        assert "L1=V2" in args_list

    @pytest.mark.asyncio
    async def test_no_cache_flag(self) -> None:
        mock_proc = _make_stream_proc()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"), no_cache=True)

        call_args = mock_exec.call_args[0]
        assert "--no-cache" in call_args

    @pytest.mark.asyncio
    async def test_no_cache_flag_absent_by_default(self) -> None:
        mock_proc = _make_stream_proc()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"))

        call_args = mock_exec.call_args[0]
        assert "--no-cache" not in call_args


# ── find_base_dockerfile ─────────────────────────────────────────────


class TestFindBaseDockerfile:
    def test_with_explicit_root(self, tmp_path: Path) -> None:
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        dockerfile = docker_dir / "Dockerfile.saber_sandbox"
        dockerfile.touch()

        result = find_base_dockerfile(saber_root=tmp_path)
        assert result == dockerfile.resolve()

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_base_dockerfile(saber_root=tmp_path)

    def test_auto_discovers_from_repo(self) -> None:
        """In the dev environment the Dockerfile exists at the repo root."""
        result = find_base_dockerfile()
        assert result.exists()
        assert result.name == "Dockerfile.saber_sandbox"


# ── Phase 3: PreflightResult ────────────────────────────────────────


class TestPreflightResult:
    def test_empty_results(self) -> None:
        pr = PreflightResult()
        assert pr.built_count == 0
        assert pr.skipped_count == 0
        assert pr.failed_count == 0
        assert pr.all_succeeded is True

    def test_counts(self) -> None:
        results = (
            ImageBuildResult(name="a", tag="a:1", action="built"),
            ImageBuildResult(name="b", tag="b:1", action="skipped"),
            ImageBuildResult(name="c", tag="c:1", action="rebuilt"),
            ImageBuildResult(name="d", tag="d:1", action="skipped"),
        )
        pr = PreflightResult(results=results, domain_slug="test")
        assert pr.built_count == 2  # built + rebuilt
        assert pr.skipped_count == 2
        assert pr.failed_count == 0
        assert pr.all_succeeded is True

    def test_all_succeeded_false_when_failed(self) -> None:
        results = (
            ImageBuildResult(name="a", tag="a:1", action="built"),
            ImageBuildResult(name="b", tag="b:1", action="failed", error="boom"),
        )
        pr = PreflightResult(results=results, domain_slug="test")
        assert pr.failed_count == 1
        assert pr.all_succeeded is False

    def test_frozen(self) -> None:
        pr = PreflightResult()
        with pytest.raises(ValidationError):
            pr.domain_slug = "other"  # type: ignore[misc]


# ── Phase 3: build_domain_images ────────────────────────────────────


def _write_eval_yaml(domain_root: Path, images: dict[str, dict[str, object]]) -> None:
    """Write a minimal eval.yaml with the given images."""
    data = {
        "slug": "test-domain",
        "name": "Test Domain",
        "description": "Test",
        "images": images,
    }
    (domain_root / "eval.yaml").write_text(yaml.dump(data))


class TestBuildDomainImages:
    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_no_images_returns_empty(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(tmp_path, {})
        result = await build_domain_images(tmp_path)
        assert result.domain_slug == "test-domain"
        assert result.results == ()
        mock_build.assert_not_called()

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_all_exist_skipped(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        # Base and domain image both exist
        mock_exists.return_value = True
        result = await build_domain_images(tmp_path)

        assert result.domain_slug == "test-domain"
        assert len(result.results) == 2  # base + web
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "skipped"
        assert actions["web"] == "skipped"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_missing_built_existing_skipped(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
                "db": {"tag": "db:1", "dockerfile": "docker/Dockerfile.db"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"

        # base exists, web exists, db doesn't
        async def exists_side_effect(tag: str) -> bool:
            return tag != "db:1"

        mock_exists.side_effect = exists_side_effect

        result = await build_domain_images(tmp_path)
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "skipped"
        assert actions["web"] == "skipped"
        assert actions["db"] == "built"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_rebuild_all(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = True  # already exist

        result = await build_domain_images(tmp_path, rebuild=RebuildMode.all())
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "rebuilt"
        assert actions["web"] == "rebuilt"
        # All rebuild calls should use no_cache=True
        for call in mock_build.call_args_list:
            assert call.kwargs.get("no_cache") is True

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_rebuild_specific(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
                "db": {"tag": "db:1", "dockerfile": "docker/Dockerfile.db"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = True  # all exist

        rebuild = RebuildMode.specific(frozenset({"web"}))
        result = await build_domain_images(tmp_path, rebuild=rebuild)
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "skipped"
        assert actions["web"] == "rebuilt"
        assert actions["db"] == "skipped"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_base_image_built_first_when_missing(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        base_df = tmp_path / "docker" / "Dockerfile.base"
        mock_find_base.return_value = base_df

        # base doesn't exist, web doesn't exist
        mock_exists.return_value = False

        result = await build_domain_images(tmp_path)
        # First result should be base
        assert result.results[0].name == "base"
        assert result.results[0].action == "built"
        assert result.results[0].tag == BASE_IMAGE_TAG

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_base_image_skipped_when_exists(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        mock_exists.return_value = True

        result = await build_domain_images(tmp_path)
        base_result = result.results[0]
        assert base_result.name == "base"
        assert base_result.action == "skipped"
        mock_find_base.assert_not_called()

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_build_failure_captured(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
                "db": {"tag": "db:1", "dockerfile": "docker/Dockerfile.db"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = False  # none exist

        # web build fails, db succeeds
        async def build_side_effect(
            tag: str,
            dockerfile: Path,
            context: Path | None = None,
            build_args: dict[str, str] | None = None,
            labels: dict[str, str] | None = None,
            no_cache: bool = False,
            status_callback: Callable[[str], None] | None = None,
        ) -> None:
            if tag == "web:1":
                raise ImageBuildError(tag="web:1", stderr="compile error", returncode=1)

        mock_build.side_effect = build_side_effect

        result = await build_domain_images(tmp_path)
        actions = {r.name: r.action for r in result.results}
        assert actions["web"] == "failed"
        assert actions["db"] == "built"
        # web should have error message
        web_result = next(r for r in result.results if r.name == "web")
        assert web_result.error is not None
        assert "compile error" in web_result.error

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_unknown_rebuild_name_warns(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        mock_exists.return_value = True

        rebuild = RebuildMode.specific(frozenset({"web", "unknown_img"}))
        with caplog.at_level(logging.WARNING, logger="saber.environments"):
            await build_domain_images(tmp_path, rebuild=rebuild)

        assert any("unknown_img" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_context_resolved_relative(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {
                    "tag": "web:1",
                    "dockerfile": "docker/Dockerfile.web",
                    "context": "docker/web_ctx",
                },
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = False

        await build_domain_images(tmp_path)

        # Find the call for the domain image (not base)
        calls = mock_build.call_args_list
        domain_call = [c for c in calls if c.kwargs.get("tag") == "web:1"]
        assert len(domain_call) == 1
        assert domain_call[0].kwargs["context"] == tmp_path / "docker" / "web_ctx"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_rebuild_specific_base_only(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = True  # all exist

        rebuild = RebuildMode.specific(frozenset({"base"}))
        result = await build_domain_images(tmp_path, rebuild=rebuild)
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "rebuilt"
        assert actions["web"] == "skipped"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_rebuild_specific_saber_sandbox_sentinel(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = True  # all exist

        rebuild = RebuildMode.specific(frozenset({"saber_sandbox"}))
        result = await build_domain_images(tmp_path, rebuild=rebuild)
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "rebuilt"

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_base_failure_short_circuits(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
                "db": {"tag": "db:1", "dockerfile": "docker/Dockerfile.db"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = False  # nothing exists

        mock_build.side_effect = ImageBuildError(tag=BASE_IMAGE_TAG, stderr="oom", returncode=137)

        result = await build_domain_images(tmp_path)
        # Short-circuit: only base result present, no domain images attempted
        assert len(result.results) == 1
        assert result.results[0].name == "base"
        assert result.results[0].action == "failed"
        assert result.all_succeeded is False
        assert result.results[0].error is not None
        assert "oom" in result.results[0].error

    @pytest.mark.asyncio
    async def test_missing_eval_yaml_graceful(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="saber.environments"):
            result = await build_domain_images(tmp_path)

        assert result.results == ()
        assert result.domain_slug == ""
        assert any("eval.yaml" in msg for msg in caplog.messages)


# ── Progress display ────────────────────────────────────────────────


class TestBuildDomainImagesProgress:
    """Tests for Rich progress bar integration in build_domain_images()."""

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_progress_display_skipped_when_not_tty(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """When stderr is not a TTY (default in pytest), _create_build_progress returns None.

        build_domain_images still works and returns correct PreflightResult.
        """
        _write_eval_yaml(
            tmp_path,
            {"web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"}},
        )
        mock_exists.return_value = True  # all exist

        # _create_build_progress returns None when not a TTY
        assert _create_build_progress(total=2) is None

        result = await build_domain_images(tmp_path)
        assert result.domain_slug == "test-domain"
        assert len(result.results) == 2  # base + web
        actions = {r.name: r.action for r in result.results}
        assert actions["base"] == "skipped"
        assert actions["web"] == "skipped"

    @pytest.mark.asyncio
    @patch("saber.environments.images.display_progress")
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_summary_displayed_after_build(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        mock_display: MagicMock,
        tmp_path: Path,
    ) -> None:
        """After build_domain_images completes, display_progress is called with a summary."""
        _write_eval_yaml(
            tmp_path,
            {"web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"}},
        )
        mock_exists.return_value = True  # all skipped

        await build_domain_images(tmp_path)

        mock_display.assert_called_once()
        summary = mock_display.call_args[0][0]
        assert "Image preflight complete" in summary
        assert "2 skipped" in summary
        # Elapsed time in seconds, e.g. "(0.0s)"
        assert "s)" in summary

    @pytest.mark.asyncio
    @patch("saber.environments.images.display_progress")
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_summary_includes_failure_count(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        mock_display: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When a build fails, the summary contains the failure count."""
        _write_eval_yaml(
            tmp_path,
            {"web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"}},
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = False

        async def build_side_effect(
            tag: str,
            dockerfile: Path,
            context: Path | None = None,
            build_args: dict[str, str] | None = None,
            labels: dict[str, str] | None = None,
            no_cache: bool = False,
            status_callback: Callable[[str], None] | None = None,
        ) -> None:
            if tag == "web:1":
                raise ImageBuildError(tag="web:1", stderr="boom", returncode=1)

        mock_build.side_effect = build_side_effect

        await build_domain_images(tmp_path)

        mock_display.assert_called_once()
        summary = mock_display.call_args[0][0]
        assert "1 failed" in summary

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    async def test_on_progress_callback_still_fires(
        self,
        mock_find_base: AsyncMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """The existing on_progress callback still receives all expected events."""
        _write_eval_yaml(
            tmp_path,
            {"web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"}},
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = False  # nothing exists, build all

        events: list[tuple[str, str, str]] = []
        callback = MagicMock(side_effect=lambda n, t, e: events.append((n, t, e)))

        await build_domain_images(tmp_path, on_progress=callback)

        # Base image events
        assert ("base", BASE_IMAGE_TAG, "building") in events
        assert ("base", BASE_IMAGE_TAG, "built") in events
        # Domain image events
        assert ("web", "web:1", "checking") in events
        assert ("web", "web:1", "building") in events
        assert ("web", "web:1", "built") in events
        assert callback.call_count == len(events)

    def test_create_build_progress_returns_none_when_not_tty(self) -> None:
        """_create_build_progress returns None when stderr is not a TTY."""
        # In pytest, stderr.isatty() is False by default
        result = _create_build_progress(total=1)
        assert result is None

    @patch("saber.environments.images._HAS_RICH", False)
    def test_create_build_progress_returns_none_when_no_rich(self) -> None:
        """_create_build_progress returns None when rich is unavailable."""
        result = _create_build_progress(total=1)
        assert result is None

    @pytest.mark.asyncio
    @patch("saber.environments.images.build_image", new_callable=AsyncMock)
    @patch("saber.environments.images.image_exists", new_callable=AsyncMock)
    @patch("saber.environments.images.find_base_dockerfile")
    @patch("saber.environments.images._create_build_progress")
    async def test_progress_active_updates_correctly(
        self,
        mock_create_progress: MagicMock,
        mock_find_base: MagicMock,
        mock_exists: AsyncMock,
        mock_build: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """When progress is active (TTY), set_status and complete_image are called."""
        mock_display = MagicMock()
        mock_display.__enter__ = MagicMock(return_value=mock_display)
        mock_display.__exit__ = MagicMock(return_value=False)
        mock_create_progress.return_value = mock_display

        _write_eval_yaml(
            tmp_path,
            {
                "web": {"tag": "web:1", "dockerfile": "docker/Dockerfile.web"},
                "db": {"tag": "db:1", "dockerfile": "docker/Dockerfile.db"},
            },
        )
        mock_find_base.return_value = tmp_path / "docker" / "Dockerfile.base"
        mock_exists.return_value = True  # all exist, but rebuild=all forces builds

        result = await build_domain_images(tmp_path, rebuild=RebuildMode.all())

        # All 3 images should be rebuilt
        assert len(result.results) == 3
        assert all(r.action == "rebuilt" for r in result.results)

        # complete_image called once per image (3 total)
        assert mock_display.complete_image.call_count == 3

        # set_description called with "Checking" or "Building" for domain images
        desc_calls = list(mock_display.set_description.call_args_list)
        desc_texts = [c[0][0] for c in desc_calls]
        assert any("web" in d for d in desc_texts)
        assert any("db" in d for d in desc_texts)


# ── _detect_build_phase ──────────────────────────────────────────────


class TestDetectBuildPhase:
    """Tests for _detect_build_phase()."""

    def test_detects_step_format(self) -> None:
        result = _detect_build_phase("Step 3/5 : RUN apt-get install -y curl")
        assert result is not None
        assert "Step 3/5" in result
        assert "RUN apt-get install" in result

    def test_detects_apt_get(self) -> None:
        result = _detect_build_phase("  Running: apt-get update")
        assert result == "installing packages\u2026"

    def test_detects_pip_install(self) -> None:
        result = _detect_build_phase("RUN pip install flask requests")
        assert result == "installing Python packages\u2026"

    def test_returns_none_for_unknown(self) -> None:
        result = _detect_build_phase("some random line of output")
        assert result is None

    def test_detects_finalize(self) -> None:
        result = _detect_build_phase("Successfully built abc123")
        assert result == "finalizing\u2026"

    def test_detects_buildkit_format(self) -> None:
        result = _detect_build_phase("#5 [stage-1 3/4] COPY . /app")
        assert result is not None
        assert "COPY . /app" in result

    def test_detects_npm_install(self) -> None:
        result = _detect_build_phase("npm install --production")
        assert result == "installing Node packages\u2026"

    def test_detects_yarn_install(self) -> None:
        result = _detect_build_phase("yarn install --frozen-lockfile")
        assert result == "installing Node packages\u2026"

    def test_detects_exporting_to_image(self) -> None:
        result = _detect_build_phase("exporting to image")
        assert result == "finalizing\u2026"

    def test_step_format_truncates_long_command(self) -> None:
        long_cmd = "RUN " + "a" * 100
        result = _detect_build_phase(f"Step 1/2 : {long_cmd}")
        assert result is not None
        assert len(result) <= 80  # "Step 1/2: " + 60 chars


# ── Build image streaming ────────────────────────────────────────────


class TestBuildImageStreaming:
    """Tests for build_image() streaming and status_callback."""

    @pytest.mark.asyncio
    async def test_status_callback_called_during_build(self) -> None:
        """status_callback receives detected build phases from stdout."""
        lines = [
            b"Step 1/3 : FROM ubuntu:22.04\n",
            b"Step 2/3 : RUN apt-get update\n",
            b"Step 3/3 : RUN pip install flask\n",
            b"Successfully built abc123def\n",
        ]
        stdout_reader = AsyncMock()
        stdout_reader.readline = AsyncMock(side_effect=[*lines, b""])
        stderr_reader = AsyncMock()
        stderr_reader.readline = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        callback = MagicMock()

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await build_image(
                tag="t:1",
                dockerfile=Path("/df"),
                context=Path("/ctx"),
                status_callback=callback,
            )

        # Should have been called for detected phases
        assert callback.call_count >= 2
        phase_args = [c[0][0] for c in callback.call_args_list]
        assert any("Step" in p for p in phase_args)
        assert any("installing" in p.lower() or "finalizing" in p.lower() for p in phase_args)

    @pytest.mark.asyncio
    async def test_build_image_works_without_callback(self) -> None:
        """build_image still works fine when no status_callback is provided."""
        stdout_reader = AsyncMock()
        stdout_reader.readline = AsyncMock(side_effect=[b"Step 1/1 : FROM ubuntu\n", b""])
        stderr_reader = AsyncMock()
        stderr_reader.readline = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"))

    @pytest.mark.asyncio
    async def test_build_image_uses_progress_plain(self) -> None:
        """Verify --progress=plain is in the docker command."""
        stdout_reader = AsyncMock()
        stdout_reader.readline = AsyncMock(return_value=b"")
        stderr_reader = AsyncMock()
        stderr_reader.readline = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"))

        call_args = mock_exec.call_args[0]
        assert "--progress=plain" in call_args

    @pytest.mark.asyncio
    async def test_build_image_streaming_failure_raises(self) -> None:
        """build_image raises ImageBuildError with stderr on failure."""
        stdout_reader = AsyncMock()
        stdout_reader.readline = AsyncMock(side_effect=[b"Step 1/2 : FROM ubuntu\n", b""])
        stderr_reader = AsyncMock()
        stderr_reader.readline = AsyncMock(side_effect=[b"error: something broke\n", b""])

        mock_proc = AsyncMock()
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(ImageBuildError) as exc_info:
                await build_image(tag="fail:1", dockerfile=Path("/df"), context=Path("/ctx"))

        assert exc_info.value.tag == "fail:1"
        assert "something broke" in exc_info.value.stderr
        assert exc_info.value.returncode == 1

    @pytest.mark.asyncio
    async def test_build_image_handles_readline_value_error(self) -> None:
        """build_image handles ValueError from readline gracefully."""
        stdout_reader = AsyncMock()
        stdout_reader.readline = AsyncMock(side_effect=[ValueError("line too long")])
        stdout_reader.read = AsyncMock(return_value=b"")
        stderr_reader = AsyncMock()
        stderr_reader.readline = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"))


# ── _BuildProgressDisplay ────────────────────────────────────────────


class TestBuildProgressDisplay:
    """Unit tests for _BuildProgressDisplay."""

    def test_enter_exit(self) -> None:
        """Context manager protocol works."""
        display = _BuildProgressDisplay(total=3)
        with display as d:
            assert d is display

    def test_set_status_updates_live(self) -> None:
        """Calling set_status updates the internal status dict."""
        display = _BuildProgressDisplay(total=2)
        with display:
            display.set_status("base", "building...")
            assert display._statuses["base"] == "building..."

    def test_complete_image_advances_bar(self) -> None:
        """Calling complete_image advances the progress bar."""
        display = _BuildProgressDisplay(total=2)
        with display:
            display.complete_image("base", "[green]\u2713[/green] rebuilt")
            assert display._statuses["base"] == "[green]\u2713[/green] rebuilt"

    def test_set_description(self) -> None:
        """Calling set_description updates the bar description."""
        display = _BuildProgressDisplay(total=1)
        with display:
            display.set_description("Checking web")
            # No assertion needed beyond no exception; verifies the method runs
