"""CLI entry point for architect-sim."""
import argparse
import sys
from .config import Config
from .loop import run_analyze, run_loop


def main():
    parser = argparse.ArgumentParser(
        description="architect-sim -- Static analysis simulator for microservice architectures"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to project root directory",
    )
    parser.add_argument(
        "--services-dir",
        help="Path to services directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        help="Path to port config file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't attempt fixes",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum fix iterations (default: 5000)",
    )
    parser.add_argument(
        "--service",
        help="Analyze a single service only",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for output files and ledger",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply fixes to disk (default: dry run analysis only)",
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Commit applied changes automatically",
    )
    parser.add_argument(
        "--apply-severity",
        choices=["critical", "warning", "info"],
        default="warning",
        help="Only apply fixes at or above this severity (default: warning)",
    )
    args = parser.parse_args()

    config = Config(
        args.root,
        services_dir=args.services_dir,
        config_file=args.config_file,
    )

    if args.analyze_only:
        grade, findings, traces, blueprints, report = run_analyze(config, args.format)
        print(report)
    else:
        report = run_loop(
            config,
            args.max_iterations,
            args.output_dir,
            apply=args.apply,
            auto_commit=args.auto_commit,
            apply_severity=args.apply_severity,
        )
        print(report)


if __name__ == "__main__":
    main()
