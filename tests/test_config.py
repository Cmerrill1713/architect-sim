def test_loads_ports_yaml(config):
    assert config.ports["user-service"] == 8080
    assert config.ports["auth-service"] == 8081

def test_resolve_port(config):
    assert config.resolve_port(8080) == "user-service"
    assert config.resolve_port(9999).startswith("unknown:")

def test_resolve_service(config):
    assert config.resolve_service("user-service") == 8080
    assert config.resolve_service("nonexistent") == 0

def test_get_service_dirs(config):
    dirs = config.get_service_dirs()
    assert "user-service" in dirs
    assert dirs["user-service"]["language"] == "go"
    assert "auth-service" in dirs
    assert dirs["auth-service"]["language"] == "python"

def test_detect_typescript(config):
    dirs = config.get_service_dirs()
    assert "web-ui" in dirs
    assert dirs["web-ui"]["language"] == "node"
