from pathlib import Path

DOCKERFILE_PATH = Path(__file__).resolve().parents[5] / "src" / "saber" / "agents" / "registry" / "firstparty" / "runtimes" / "docker" / "Dockerfile.hyenas"


class TestHyenasDockerfile:
    def test_dockerfile_exists(self):
        assert DOCKERFILE_PATH.exists()

    def test_from_node_base(self):
        content = DOCKERFILE_PATH.read_text()
        assert "FROM node:" in content

    def test_installs_ripgrep(self):
        content = DOCKERFILE_PATH.read_text()
        assert "ripgrep" in content

    def test_installs_git(self):
        content = DOCKERFILE_PATH.read_text()
        assert "git" in content

    def test_creates_workspace_dir(self):
        content = DOCKERFILE_PATH.read_text()
        assert "/workspace" in content

    def test_creates_output_dir(self):
        content = DOCKERFILE_PATH.read_text()
        assert "/output" in content

    def test_workdir_app(self):
        content = DOCKERFILE_PATH.read_text()
        assert "WORKDIR /app" in content

    def test_keepalive_cmd(self):
        content = DOCKERFILE_PATH.read_text()
        assert "tail" in content
