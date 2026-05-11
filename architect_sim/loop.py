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
from .extractors.llm_calls import extract_llm_calls_all_languages, format_llm_report
from .runtime.log_ingestion import auto_discover_logs, ingest_logs, correlate_with_findings, format_runtime_report
from .profiling.hardware import detect_hardware, recommend_models, format_hardware_report
from .scoring.history import ScoreHistory
from .reporting.markdown_report import generate_report
from .reporting.json_report import generate_json
from .reporting.recommendations import generate_recommendations
from .reporting.ledger import Ledger
from .research.tech_upgrades import scan_for_technologies, format_tech_report


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

    # Phase 4: LLM call extraction
    print("Phase 4: Extracting LLM call sites...", file=sys.stderr)
    start = time.time()
    all_llm_calls = []
    for name, bp in blueprints.items():
        calls = extract_llm_calls_all_languages(name, bp.source_dir)
        all_llm_calls.extend(calls)
    llm_time = time.time() - start
    task_counts = {}
    for c in all_llm_calls:
        task_counts[c.task_type] = task_counts.get(c.task_type, 0) + 1
    print(f"  {len(all_llm_calls)} LLM call sites across {len(set(c.service_name for c in all_llm_calls))} services ({llm_time:.1f}s)", file=sys.stderr)
    if task_counts:
        print(f"  Tasks: {task_counts}", file=sys.stderr)

    # Phase 5: Runtime log analysis
    print("Phase 5: Analyzing runtime logs...", file=sys.stderr)
    start = time.time()
    log_paths = auto_discover_logs(str(config.root))
    runtime_events = []
    correlations = []
    if log_paths:
        runtime_events = ingest_logs(log_paths, since_hours=24)
        correlations = correlate_with_findings(runtime_events, findings, config)
        confirmed = sum(1 for c in correlations if c.severity_boost == "confirmed")
        print(f"  {len(runtime_events)} events from {len(log_paths)} logs, {confirmed} findings confirmed by runtime ({time.time() - start:.1f}s)", file=sys.stderr)
    else:
        print(f"  No log files found ({time.time() - start:.1f}s)", file=sys.stderr)

    # Phase 6: Hardware profiling
    print("Phase 6: Profiling hardware...", file=sys.stderr)
    start = time.time()
    hardware = detect_hardware()
    model_recs = recommend_models(hardware, task_types=task_counts if task_counts else None)
    print(f"  {hardware.cpu_model}, {hardware.ram_total_gb:.0f}GB, {len(model_recs)} models fit ({time.time() - start:.1f}s)", file=sys.stderr)

    # Phase 7: Generate recommendations
    print("Phase 7: Generating action plan...", file=sys.stderr)
    action_plan = generate_recommendations(
        grade, findings, blueprints,
        llm_calls=all_llm_calls,
        runtime_events=runtime_events,
        correlations=correlations,
        hardware=hardware,
        model_recs=model_recs,
        temporal_profiles=temporal_profiles,
    )
    rec_count = action_plan.count("### ")
    print(f"  {rec_count} prioritized recommendations", file=sys.stderr)

    # Phase 8: Technology upgrade scan
    print("Phase 8: Scanning for technology upgrades...", file=sys.stderr)
    start = time.time()
    tech_upgrades = scan_for_technologies(str(config.root), blueprints)
    print(f"  {len(tech_upgrades)} upgrade recommendations ({time.time() - start:.1f}s)", file=sys.stderr)

    if output_format == "json":
        report = generate_json(grade, findings, traces, blueprints)
    else:
        # Action plan + tech upgrades go FIRST
        report = action_plan + "\n\n---\n\n"
        if tech_upgrades:
            report += format_tech_report(tech_upgrades) + "\n\n---\n\n"
        report += generate_report(grade, findings, traces, blueprints)
        report += "\n\n" + format_temporal_report(temporal_profiles)
        if all_llm_calls:
            report += "\n\n" + format_llm_report(all_llm_calls)
        if runtime_events:
            report += "\n\n" + format_runtime_report(correlations, runtime_events)
        report += "\n\n" + format_hardware_report(hardware, model_recs)

    # Record score history
    try:
        history = ScoreHistory(str(config.root))
        history.record(grade, findings, blueprints)
        trend = history.format_trend()
        if trend and output_format != "json":
            report += "\n\n" + trend
    except Exception:
        pass  # Score history is best-effort

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
