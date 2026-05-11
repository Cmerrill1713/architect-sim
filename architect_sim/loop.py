"""Main autonomous simulation loop.

Extract -> Simulate -> Grade -> Diagnose -> Fix -> Re-Extract -> Re-Grade -> iterate
"""
import sys
import time
from .config import Config
from .models import ServiceBlueprint, IterationRecord
from .extractors.go_routes import extract_go_routes
from .extractors.go_calls import extract_go_calls
from .extractors.rust_routes import extract_rust_routes, extract_rust_calls
from .extractors.python_routes import extract_python_routes, extract_python_calls
from .extractors.temporal import (
    enrich_blueprints_temporal, scan_docs_for_references,
    find_doc_drift, format_temporal_report,
)
from .simulation.contract_checker import check_contracts, check_dependency_drift, check_circular_deps
from .simulation.flow_tracer import trace_all_flows
from .grading.scorer import score_system
from .reporting.markdown_report import generate_report
from .reporting.json_report import generate_json
from .reporting.ledger import Ledger


def extract_all(config: Config) -> dict:
    """Phase 1: Extract service blueprints from all source code.

    Returns:
        {service_name: ServiceBlueprint}
    """
    blueprints = {}
    service_dirs = config.get_service_dirs()

    for svc_name, info in service_dirs.items():
        lang = info["language"]
        svc_dir = info["path"]
        port = config.resolve_service(svc_name)

        bp = ServiceBlueprint(
            name=svc_name,
            port=port,
            language=lang,
            source_dir=svc_dir,
        )

        if lang == "go":
            bp.endpoints = extract_go_routes(svc_name, port, svc_dir)
            bp.outbound_calls = extract_go_calls(svc_name, svc_dir, config)
        elif lang == "rust":
            bp.endpoints = extract_rust_routes(svc_name, port, svc_dir)
            bp.outbound_calls = extract_rust_calls(svc_name, svc_dir, config)
        elif lang == "python":
            bp.endpoints = extract_python_routes(svc_name, port, svc_dir)
            bp.outbound_calls = extract_python_calls(svc_name, svc_dir, config)

        if bp.endpoints or bp.outbound_calls:
            blueprints[svc_name] = bp

    return blueprints


def simulate_all(blueprints: dict, config: Config) -> tuple:
    """Phase 2: Run all simulations.

    Returns:
        (findings: list[Finding], traces: list[FlowTrace])
    """
    findings = []

    # Contract checks
    findings.extend(check_contracts(blueprints, config))

    # Dependency drift
    findings.extend(check_dependency_drift(blueprints, config))

    # Circular dependencies
    findings.extend(check_circular_deps(blueprints))

    # Auto-discovered flow traces
    traces, flow_findings = trace_all_flows(blueprints, config)
    findings.extend(flow_findings)

    # Deduplicate findings by (type, service, endpoint)
    seen = set()
    unique_findings = []
    for f in findings:
        key = (f.finding_type, f.service, f.endpoint, f.details)
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    return unique_findings, traces


def run_analyze(config: Config, output_format: str = "markdown") -> tuple:
    """Run extract + simulate + grade (read-only analysis).

    Returns:
        (grade, findings, traces, blueprints, report_str)
    """
    print("Phase 1: Extracting service blueprints...", file=sys.stderr)
    start = time.time()
    blueprints = extract_all(config)
    extract_time = time.time() - start
    print(f"  Extracted {len(blueprints)} services, "
          f"{sum(len(bp.endpoints) for bp in blueprints.values())} endpoints, "
          f"{sum(len(bp.outbound_calls) for bp in blueprints.values())} calls "
          f"({extract_time:.1f}s)", file=sys.stderr)

    # Temporal enrichment (git history)
    print("Phase 1b: Enriching with git history...", file=sys.stderr)
    start = time.time()
    temporal_profiles = enrich_blueprints_temporal(blueprints, str(config.root))
    temporal_time = time.time() - start
    active = sum(1 for p in temporal_profiles.values() if p.staleness == "active")
    stale = sum(1 for p in temporal_profiles.values() if p.staleness in ("stale", "abandoned"))
    print(f"  {active} active, {stale} stale/abandoned ({temporal_time:.1f}s)", file=sys.stderr)

    # Doc cross-reference
    print("Phase 1c: Scanning documentation...", file=sys.stderr)
    start = time.time()
    doc_refs = scan_docs_for_references(str(config.root), blueprints, config)
    doc_findings = find_doc_drift(doc_refs, blueprints, config)
    doc_time = time.time() - start
    print(f"  {len(doc_refs)} references, {len(doc_findings)} drift issues ({doc_time:.1f}s)", file=sys.stderr)

    print("Phase 2: Running simulations...", file=sys.stderr)
    start = time.time()
    findings, traces = simulate_all(blueprints, config)
    findings.extend(doc_findings)
    sim_time = time.time() - start
    print(f"  Found {len(findings)} issues ({sim_time:.1f}s)", file=sys.stderr)

    print("Phase 3: Grading system...", file=sys.stderr)
    grade = score_system(blueprints, findings, traces, config)
    print(f"  Grade: {grade.grade} ({grade.overall:.1f}/100)", file=sys.stderr)

    if output_format == "json":
        report = generate_json(grade, findings, traces, blueprints)
    else:
        report = generate_report(grade, findings, traces, blueprints)
        report += "\n\n" + format_temporal_report(temporal_profiles)

    return grade, findings, traces, blueprints, report


def run_loop(config: Config, max_iterations: int = 50, output_dir: str = None):
    """Run the full autonomous simulation loop.

    Extract -> Simulate -> Grade -> Diagnose -> Fix -> Re-Grade -> iterate
    """
    if output_dir is None:
        output_dir = str(config.root / "architect-sim-output")

    ledger = Ledger(output_dir)

    # Initial analysis
    grade, findings, traces, blueprints, report = run_analyze(config)

    print(f"\nInitial Grade: {grade.grade} ({grade.overall:.1f}/100)", file=sys.stderr)
    print(f"Critical: {sum(1 for f in findings if f.severity == 'critical')}", file=sys.stderr)
    print(f"Warning: {sum(1 for f in findings if f.severity == 'warning')}", file=sys.stderr)
    print(f"Info: {sum(1 for f in findings if f.severity == 'info')}", file=sys.stderr)

    # Sort findings: critical first, then by fixability (auto > needs_context > needs_user)
    severity_order = {"critical": 0, "warning": 1, "info": 2}

    auto_fixable = [
        f for f in findings
        if f.fixability == "auto" and f.severity in ("critical", "warning")
    ]
    auto_fixable.sort(key=lambda f: (severity_order.get(f.severity, 9)))

    needs_user = [f for f in findings if f.fixability == "needs_user"]

    print(f"\nAuto-fixable: {len(auto_fixable)}", file=sys.stderr)
    print(f"Needs user input: {len(needs_user)}", file=sys.stderr)

    # Run fast in-memory fix iterations
    from .fixing.fast_engine import run_fast_iterations, format_fix_report

    print(f"\nPhase 4: Running fast fix iterations (up to {max_iterations})...", file=sys.stderr)
    fix_result = run_fast_iterations(blueprints, findings, config, max_iterations)

    print(f"  {fix_result.iterations_run} iterations in {fix_result.elapsed_ms:.0f}ms "
          f"({fix_result.iterations_per_second:.0f} iter/s)", file=sys.stderr)
    print(f"  Accepted: {len(fix_result.fixes_accepted)} fixes", file=sys.stderr)
    print(f"  Score: {fix_result.initial_grade.grade} ({fix_result.initial_grade.overall:.1f}) -> "
          f"{fix_result.final_grade.grade} ({fix_result.final_grade.overall:.1f})", file=sys.stderr)

    if fix_result.needs_user:
        print(f"\n  Needs your decision: {len(fix_result.needs_user)} issues", file=sys.stderr)

    return format_fix_report(fix_result)
