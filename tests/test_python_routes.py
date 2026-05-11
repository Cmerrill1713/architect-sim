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
