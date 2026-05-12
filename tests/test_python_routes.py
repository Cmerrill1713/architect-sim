from architect_sim.extractors.python_routes import extract_python_routes, extract_python_calls

def test_extracts_fastapi_routes(tmp_project):
    routes = extract_python_routes("auth-service", 8081,
                                    str(tmp_project / "services" / "auth-service"))
    paths = {(r.method, r.path) for r in routes}
    assert ("GET", "/health") in paths
    assert ("POST", "/auth/login") in paths
    assert ("GET", "/auth/verify") in paths

def test_extracts_outbound_calls(tmp_project, config):
    calls = extract_python_calls("auth-service",
                                  str(tmp_project / "services" / "auth-service"), config)
    assert len(calls) >= 1
    assert any(c.callee_port == 8080 for c in calls)


def test_skips_python_self_test_probe_urls(tmp_path, config):
    service_dir = tmp_path / "services" / "probe-service"
    service_dir.mkdir(parents=True)
    (service_dir / "probe.py").write_text('''
def self_test():
    probe_http("http://127.0.0.1:19999/health", "test_service")

def runtime_probe():
    probe_http("http://127.0.0.1:8100/health", "agi_core")
''')

    calls = extract_python_calls("probe-service", str(service_dir), config)
    ports = {c.callee_port for c in calls}
    assert 19999 not in ports
    assert 8100 in ports
