from architect_sim.simulation.contract_checker import check_contracts, _base_name, _fuzzy_match

def test_base_name():
    assert _base_name("user-service-go") == "user-service"
    assert _base_name("rag-gateway-rust") == "rag-gateway"
    assert _base_name("auth-py") == "auth"
    assert _base_name("plain-name") == "plain-name"

def test_fuzzy_match():
    # _base_name strips -go/-rust/-py/-python/-service suffixes
    # "agi-core-go" and "agi-core" both base to "agi-core"
    assert _fuzzy_match("agi-core-go", "agi-core")
    assert _fuzzy_match("rag-gateway", "rag-gateway-rust")
    assert not _fuzzy_match("user-service", "auth-service")

def test_finds_phantom_calls(config):
    from architect_sim.loop import extract_all, simulate_all
    blueprints = extract_all(config)
    findings, _ = simulate_all(blueprints, config)
    # auth-service calls user-service:8080/api/users/123
    # Dynamic route matching should resolve /api/users/:id, so this is not a
    # phantom call anymore.
    phantoms = [f for f in findings if f.finding_type == "phantom_call"]
    assert not any("/api/users/123" in f.endpoint for f in phantoms)
