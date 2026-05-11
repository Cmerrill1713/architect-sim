from architect_sim.extractors.ts_routes import extract_ts_routes, extract_ts_calls

def test_extracts_express_routes(tmp_project):
    routes = extract_ts_routes("web-ui", 3000,
                                str(tmp_project / "services" / "web-ui"))
    paths = {(r.method, r.path) for r in routes}
    assert ("GET", "/health") in paths
    assert ("POST", "/api/chat") in paths

def test_extracts_fetch_calls(tmp_project, config):
    calls = extract_ts_calls("web-ui",
                              str(tmp_project / "services" / "web-ui"), config)
    assert len(calls) >= 1
    assert any(c.callee_port == 8080 for c in calls)
