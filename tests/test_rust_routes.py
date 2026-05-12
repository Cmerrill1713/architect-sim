from architect_sim.extractors.rust_routes import extract_rust_calls


def test_skips_rust_integration_tests(tmp_path, config):
    service_dir = tmp_path / "services" / "rag-gateway-rust"
    tests_dir = service_dir / "tests"
    src_dir = service_dir / "src"
    tests_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    (tests_dir / "cache_bust_test.rs").write_text('''
#[tokio::test]
async fn cache_busts() {
    reqwest::Client::new().post("http://localhost:8089/events").send().await.unwrap();
}
''')
    (src_dir / "main.rs").write_text('''
async fn health_check() {
    reqwest::Client::new().get("http://localhost:8100/health").send().await.unwrap();
}
''')

    calls = extract_rust_calls("rag-gateway-rust", str(service_dir), config)
    ports = {c.callee_port for c in calls}
    assert 8089 not in ports
    assert 8100 in ports
