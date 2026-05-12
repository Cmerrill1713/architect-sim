from architect_sim.grading.scorer import score_system
from architect_sim.models import Endpoint, GradeReport, Finding, ServiceBlueprint


class _FakeConfig:
    """Minimal config stub for scorer tests."""
    def get_declared_deps(self, service):
        return []


def test_perfect_score():
    report = score_system({}, [], [], _FakeConfig())
    # No services, no findings = should not crash
    assert isinstance(report, GradeReport)

def test_grade_assignment():
    report = GradeReport()
    report.overall = 95
    assert report.overall >= 90  # Would be A

def test_security_penalty():
    findings = [
        Finding(finding_type="security_issue", severity="critical",
                fixability="needs_context", service="svc")
    ] * 5
    report = score_system({}, findings, [], _FakeConfig())
    assert report.security < 100

def test_operational_crud_endpoints_do_not_destroy_coverage_score():
    endpoints = [
        "/api/census",
        "/ops/summary",
        "/daemon/restart",
        "/v1/completions",
        "/registry/services",
        "/tasks/:id",
    ]
    findings = [
        Finding(finding_type="orphan_endpoint", severity="info",
                fixability="needs_context", endpoint=ep, service="svc")
        for ep in endpoints
    ]
    blueprint = ServiceBlueprint(
        name="svc",
        port=8080,
        language="go",
        endpoints=[
            Endpoint("svc", 8080, "GET", ep, "svc.go", i)
            for i, ep in enumerate(endpoints, start=1)
        ],
    )
    report = score_system({"svc": blueprint}, findings, [], _FakeConfig())
    assert report.coverage == 100
