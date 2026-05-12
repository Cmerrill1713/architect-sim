from architect_sim.extractors.go_calls import extract_go_calls


class DummyConfig:
    def resolve_service(self, service_name):
        return {"test-service": 9999}.get(service_name, 0)

    def resolve_port(self, port):
        return {8196: "service-manager-rust", 9999: "test-service"}.get(port, f"unknown:{port}")


def _write_go(tmp_path, body):
    service_dir = tmp_path / "service"
    service_dir.mkdir()
    (service_dir / "main.go").write_text(body)
    return service_dir


def test_extracts_concatenated_hardcoded_url_path(tmp_path):
    service_dir = _write_go(
        tmp_path,
        """
package main

func restart(action Action) {
    client.Post("http://localhost:8196/services/" + action.Target + "/restart", "application/json", nil)
}
""",
    )

    calls = extract_go_calls("test-service", str(service_dir), DummyConfig())

    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].callee_service == "service-manager-rust"
    assert calls[0].path == "/services/:target/restart"


def test_plain_hardcoded_url_path_is_unchanged(tmp_path):
    service_dir = _write_go(
        tmp_path,
        """
package main

func health() {
    http.Get("http://localhost:8196/health")
}
""",
    )

    calls = extract_go_calls("test-service", str(service_dir), DummyConfig())

    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert calls[0].path == "/health"


def test_dynamic_suffix_without_later_literal_becomes_param(tmp_path):
    service_dir = _write_go(
        tmp_path,
        """
package main

func service(action Action) {
    http.Get("http://localhost:8196/services/" + action.Target)
}
""",
    )

    calls = extract_go_calls("test-service", str(service_dir), DummyConfig())

    assert len(calls) == 1
    assert calls[0].path == "/services/:target"
