from architect_sim.extractors.go_routes import extract_go_routes

def test_extracts_gin_routes(tmp_project):
    routes = extract_go_routes("user-service", 8080,
                               str(tmp_project / "services" / "user-service"))
    paths = {(r.method, r.path) for r in routes}
    assert ("GET", "/health") in paths
    assert ("POST", "/api/users") in paths
    assert ("GET", "/api/users/:id") in paths

def test_extracts_gin_groups(tmp_project):
    routes = extract_go_routes("user-service", 8080,
                               str(tmp_project / "services" / "user-service"))
    paths = {(r.method, r.path) for r in routes}
    assert ("GET", "/api/v1/items") in paths
    assert ("POST", "/api/v1/items") in paths

def test_sets_framework(tmp_project):
    routes = extract_go_routes("user-service", 8080,
                               str(tmp_project / "services" / "user-service"))
    assert all(r.framework == "gin" for r in routes)

def test_records_file_and_line(tmp_project):
    routes = extract_go_routes("user-service", 8080,
                               str(tmp_project / "services" / "user-service"))
    for r in routes:
        assert r.file_path.endswith(".go")
        assert r.line_number > 0
