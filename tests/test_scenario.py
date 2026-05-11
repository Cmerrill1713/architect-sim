from architect_sim.simulator.scenario_engine import PRESET_SCENARIOS

def test_presets_exist():
    assert len(PRESET_SCENARIOS) >= 5
    assert "kill_dead_ports" in PRESET_SCENARIOS
    assert "full_resilience" in PRESET_SCENARIOS

def test_preset_has_required_fields():
    for name, scenario in PRESET_SCENARIOS.items():
        assert scenario.name
        assert scenario.description
        assert scenario.effort in ("small", "medium", "large")
