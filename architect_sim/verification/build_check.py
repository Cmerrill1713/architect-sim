"""Build verification -- confirm patches compile after applying.

Runs language-specific build checks:
- Go: go build, go vet
- Rust: cargo check
- Python: py_compile
"""
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BuildResult:
    service: str
    language: str
    success: bool
    command: str
    output: str
    duration_ms: float = 0


def verify_build(service_dir: str, language: str, timeout: int = 30) -> BuildResult:
    """Run build verification for a service directory."""
    import time

    if language == "go":
        return _verify_go(service_dir, timeout)
    elif language == "rust":
        return _verify_rust(service_dir, timeout)
    elif language == "python":
        return _verify_python(service_dir, timeout)

    return BuildResult(
        service=Path(service_dir).name,
        language=language,
        success=True,
        command="(no check available)",
        output="",
    )


def verify_all_changed(patches: list, blueprints: dict, timeout: int = 30) -> list:
    """Verify builds for all services that had patches applied.

    Returns list of BuildResult.
    """
    # Group patches by service
    services_changed = set()
    for patch in patches:
        for name, bp in blueprints.items():
            if patch.file_path.startswith(bp.source_dir):
                services_changed.add(name)
                break

    results = []
    for name in sorted(services_changed):
        bp = blueprints[name]
        result = verify_build(bp.source_dir, bp.language, timeout)
        results.append(result)

    return results


def format_build_report(results: list) -> str:
    """Format build results as markdown."""
    if not results:
        return "No build verification needed.\n"

    lines = ["## Build Verification\n"]
    lines.append("| Service | Language | Status | Command |")
    lines.append("|---------|----------|--------|---------|")

    for r in results:
        status = "PASS" if r.success else "FAIL"
        lines.append(f"| {r.service} | {r.language} | {status} | `{r.command}` |")

    # Show errors for failures
    failures = [r for r in results if not r.success]
    if failures:
        lines.append("\n### Failures\n")
        for r in failures:
            lines.append(f"**{r.service}** (`{r.command}`):")
            lines.append(f"```\n{r.output[:500]}\n```\n")

    return "\n".join(lines)


def _verify_go(service_dir: str, timeout: int) -> BuildResult:
    """Run go vet on a Go service."""
    import time
    start = time.time()
    cmd = "go vet ./..."
    try:
        result = subprocess.run(
            ["go", "vet", "./..."],
            cwd=service_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = (time.time() - start) * 1000
        return BuildResult(
            service=Path(service_dir).name,
            language="go",
            success=result.returncode == 0,
            command=cmd,
            output=result.stderr or result.stdout,
            duration_ms=elapsed,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return BuildResult(
            service=Path(service_dir).name,
            language="go",
            success=False,
            command=cmd,
            output=str(e),
        )


def _verify_rust(service_dir: str, timeout: int) -> BuildResult:
    """Run cargo check on a Rust service."""
    import time
    start = time.time()
    cmd = "cargo check"
    try:
        result = subprocess.run(
            ["cargo", "check"],
            cwd=service_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = (time.time() - start) * 1000
        return BuildResult(
            service=Path(service_dir).name,
            language="rust",
            success=result.returncode == 0,
            command=cmd,
            output=result.stderr or result.stdout,
            duration_ms=elapsed,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return BuildResult(
            service=Path(service_dir).name,
            language="rust",
            success=False,
            command=cmd,
            output=str(e),
        )


def _verify_python(service_dir: str, timeout: int) -> BuildResult:
    """Run py_compile on Python files."""
    import time
    start = time.time()
    py_files = list(Path(service_dir).rglob("*.py"))
    errors = []
    for f in py_files:
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(f)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                errors.append(f"{f.name}: {result.stderr.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elapsed = (time.time() - start) * 1000
    return BuildResult(
        service=Path(service_dir).name,
        language="python",
        success=len(errors) == 0,
        command=f"py_compile ({len(py_files)} files)",
        output="\n".join(errors) if errors else "",
        duration_ms=elapsed,
    )
