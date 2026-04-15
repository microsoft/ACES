import yaml
from pathlib import Path

COMPOSE_PATH = Path(__file__).resolve().parents[5] / "src" / "saber" / "agents" / "registry" / "firstparty" / "runtimes" / "compose" / "hyenas.sandbox.compose.yml"


class TestHyenasCompose:
    def test_compose_file_exists(self):
        assert COMPOSE_PATH.exists()

    def test_yaml_is_valid(self):
        content = COMPOSE_PATH.read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_has_default_service(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        assert "default" in parsed["services"]

    def test_has_volumes(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        assert "workspace" in parsed.get("volumes", {})
        assert "output" in parsed.get("volumes", {})

    def test_has_networks(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        assert "networks" in parsed

    def test_service_has_image(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        svc = parsed["services"]["default"]
        assert "image" in svc

    def test_service_has_init(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        svc = parsed["services"]["default"]
        assert svc.get("init") is True

    def test_service_has_resource_limits(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        svc = parsed["services"]["default"]
        limits = svc["deploy"]["resources"]["limits"]
        assert "cpus" in limits
        assert "memory" in limits

    def test_keepalive_command(self):
        parsed = yaml.safe_load(COMPOSE_PATH.read_text())
        svc = parsed["services"]["default"]
        assert "tail" in str(svc.get("command", ""))
