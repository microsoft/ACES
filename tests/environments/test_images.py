"""Tests for saber.environments.images – Phases 2 & 3."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


class TestBuildImage:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

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
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
        mock_proc.returncode = 1

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with pytest.raises(ImageBuildError) as exc_info:
                await build_image(tag="fail:1", dockerfile=Path("/df"), context=Path("/ctx"))

        assert exc_info.value.tag == "fail:1"
        assert exc_info.value.stderr == "error msg"
        assert exc_info.value.returncode == 1

    @pytest.mark.asyncio
    async def test_default_context_is_dockerfile_parent(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

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
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

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
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch(
            "saber.environments.images.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await build_image(tag="t:1", dockerfile=Path("/df"), context=Path("/ctx"), no_cache=True)

        call_args = mock_exec.call_args[0]
        assert "--no-cache" in call_args

    @pytest.mark.asyncio
    async def test_no_cache_flag_absent_by_default(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

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
