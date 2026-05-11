"""Temporal analysis -- git history enrichment for service blueprints.

Adds last-modified dates, commit frequency, and staleness classification
to each service. This makes rules smarter:
  - Code untouched for 30+ days is more likely dead
  - Actively developed services get benefit of the doubt
  - Docs that reference removed services indicate drift
"""
import subprocess
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TemporalProfile:
    """Temporal metadata for a service or file."""
    last_modified: datetime = None
    first_commit: datetime = None
    commit_count_30d: int = 0      # Commits in last 30 days
    commit_count_total: int = 0
    days_since_modified: int = 0
    staleness: str = "unknown"     # "active", "recent", "stale", "abandoned"
    last_author: str = ""
    top_contributors: list = field(default_factory=list)


@dataclass
class DocReference:
    """A reference found in documentation."""
    doc_path: str
    line_number: int
    service_name: str             # Service mentioned
    port: int = 0                 # Port mentioned
    endpoint: str = ""            # Endpoint mentioned
    context: str = ""             # Surrounding text


def enrich_blueprints_temporal(blueprints: dict, root: str) -> dict:
    """Add temporal profiles to each service blueprint.

    Returns:
        {service_name: TemporalProfile}
    """
    profiles = {}
    root_path = Path(root)

    for name, bp in blueprints.items():
        svc_dir = bp.source_dir
        if not svc_dir or not Path(svc_dir).exists():
            profiles[name] = TemporalProfile(staleness="missing")
            continue

        # Get relative path for git
        try:
            rel_path = str(Path(svc_dir).relative_to(root_path))
        except ValueError:
            rel_path = svc_dir

        profile = TemporalProfile()

        # Last modified date
        last_mod = _git_last_modified(root, rel_path)
        if last_mod:
            profile.last_modified = last_mod
            profile.days_since_modified = (datetime.now(timezone.utc) - last_mod).days

        # First commit date
        first = _git_first_commit(root, rel_path)
        if first:
            profile.first_commit = first

        # Commit count in last 30 days
        profile.commit_count_30d = _git_commit_count(root, rel_path, days=30)
        profile.commit_count_total = _git_commit_count(root, rel_path, days=None)

        # Last author
        profile.last_author = _git_last_author(root, rel_path)

        # Classify staleness
        if profile.days_since_modified <= 7 or profile.commit_count_30d >= 5:
            profile.staleness = "active"
        elif profile.days_since_modified <= 30:
            profile.staleness = "recent"
        elif profile.days_since_modified <= 90:
            profile.staleness = "stale"
        else:
            profile.staleness = "abandoned"

        profiles[name] = profile

    return profiles


def scan_docs_for_references(root: str, blueprints: dict, config) -> list:
    """Scan documentation for service/port/endpoint references.

    Finds:
    - Docs that mention services not in the codebase (doc drift)
    - Docs that mention ports not in config
    - Docs with outdated endpoint references

    Returns:
        list[DocReference]
    """
    references = []
    root_path = Path(root)

    # Look for docs in common locations
    doc_dirs = []
    for candidate in ["docs", "doc", "documentation"]:
        d = root_path / candidate
        if d.exists():
            doc_dirs.append(d)
    # Also check root for markdown files
    doc_dirs.append(root_path)

    # Build lookup sets
    known_services = set(blueprints.keys())
    known_ports = set(config.ports.values())

    # Port pattern
    port_re = re.compile(r'(?:port|PORT|localhost|127\.0\.0\.1)[:\s]+(\d{4,5})')
    # Service name pattern (look for hyphenated names with language suffixes)
    service_re = re.compile(r'\b([a-z][\w-]*(?:-go|-rust|-py|-service))\b')
    # Endpoint pattern
    endpoint_re = re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}:*%-]+)')

    for docs_dir in doc_dirs:
        pattern = "*.md" if docs_dir == root_path else "**/*.md"
        for md_file in docs_dir.glob(pattern):
            try:
                content = md_file.read_text(errors="replace")
            except (OSError, IOError):
                continue

            lines = content.split("\n")
            file_str = str(md_file)

            for i, line in enumerate(lines, 1):
                for match in port_re.finditer(line):
                    port = int(match.group(1))
                    if 1024 < port < 65535:
                        references.append(DocReference(
                            doc_path=file_str,
                            line_number=i,
                            service_name=config.port_to_service.get(port, f"unknown:{port}"),
                            port=port,
                            context=line.strip()[:120],
                        ))

                for match in service_re.finditer(line):
                    svc_name = match.group(1)
                    references.append(DocReference(
                        doc_path=file_str,
                        line_number=i,
                        service_name=svc_name,
                        context=line.strip()[:120],
                    ))

                for match in endpoint_re.finditer(line):
                    endpoint = match.group(0)
                    references.append(DocReference(
                        doc_path=file_str,
                        line_number=i,
                        service_name="",
                        endpoint=endpoint,
                        context=line.strip()[:120],
                    ))

    return references


def find_doc_drift(doc_refs: list, blueprints: dict, config) -> list:
    """Find documentation that references things that don't exist in code.

    Returns list of Finding objects.
    """
    from ..models import Finding

    findings = []
    known_services = set(blueprints.keys()) | set(config.ports.keys())
    known_ports = set(config.ports.values())

    # Build endpoint set
    all_endpoints = set()
    for bp in blueprints.values():
        for ep in bp.endpoints:
            all_endpoints.add(f"{ep.method} {ep.path}")

    seen = set()

    for ref in doc_refs:
        # Check for ports not in system
        if ref.port and ref.port not in known_ports:
            key = (ref.doc_path, ref.port)
            if key not in seen:
                seen.add(key)
                findings.append(Finding(
                    finding_type="doc_drift",
                    severity="info",
                    fixability="needs_context",
                    service="docs",
                    endpoint=f"port {ref.port}",
                    file_path=ref.doc_path,
                    line_number=ref.line_number,
                    details=f"Doc references port {ref.port} which is not registered in config: {ref.context}",
                    suggested_fix="Update documentation or register the port",
                ))

        # Check for service names not in system
        if ref.service_name and ref.service_name.endswith(("-go", "-rust", "-py", "-service")):
            if ref.service_name not in known_services:
                key = (ref.doc_path, ref.service_name)
                if key not in seen:
                    seen.add(key)
                    findings.append(Finding(
                        finding_type="doc_drift",
                        severity="info",
                        fixability="needs_context",
                        service="docs",
                        endpoint=ref.service_name,
                        file_path=ref.doc_path,
                        line_number=ref.line_number,
                        details=f"Doc references service '{ref.service_name}' which has no blueprint: {ref.context}",
                        suggested_fix="Update documentation to reflect current service names",
                    ))

    return findings


def format_temporal_report(profiles: dict) -> str:
    """Format temporal analysis as markdown."""
    lines = []
    lines.append("## Service Activity Timeline\n")
    lines.append("| Service | Last Modified | Days Ago | Commits (30d) | Status |")
    lines.append("|---------|--------------|----------|---------------|--------|")

    sorted_profiles = sorted(
        profiles.items(),
        key=lambda x: x[1].last_modified or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    for name, p in sorted_profiles:
        last_mod = p.last_modified.strftime("%Y-%m-%d") if p.last_modified else "never"
        status_icon = {
            "active": "ACTIVE",
            "recent": "RECENT",
            "stale": "STALE",
            "abandoned": "ABANDONED",
            "missing": "MISSING",
            "unknown": "?",
        }.get(p.staleness, "?")
        lines.append(f"| {name} | {last_mod} | {p.days_since_modified} | {p.commit_count_30d} | {status_icon} |")

    status_counts = {}
    for p in profiles.values():
        status_counts[p.staleness] = status_counts.get(p.staleness, 0) + 1

    lines.append(f"\n**Active:** {status_counts.get('active', 0)} | "
                 f"**Recent:** {status_counts.get('recent', 0)} | "
                 f"**Stale:** {status_counts.get('stale', 0)} | "
                 f"**Abandoned:** {status_counts.get('abandoned', 0)}")

    return "\n".join(lines)


# ============================================================
# Git helpers (subprocess, no dependencies)
# ============================================================

def _git_cmd(root: str, args: list, timeout: int = 5) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _git_last_modified(root: str, path: str) -> datetime:
    """Get the last modified date for a path."""
    output = _git_cmd(root, ["log", "-1", "--format=%aI", "--", path])
    if output:
        try:
            return datetime.fromisoformat(output)
        except ValueError:
            pass
    return None


def _git_first_commit(root: str, path: str) -> datetime:
    """Get the first commit date for a path."""
    output = _git_cmd(root, ["log", "--reverse", "--format=%aI", "--", path], timeout=10)
    if output:
        first_line = output.split("\n")[0]
        try:
            return datetime.fromisoformat(first_line)
        except ValueError:
            pass
    return None


def _git_commit_count(root: str, path: str, days: int = None) -> int:
    """Count commits touching a path, optionally within N days."""
    args = ["log", "--oneline", "--", path]
    if days:
        args = ["log", "--oneline", f"--since={days} days ago", "--", path]
    output = _git_cmd(root, args, timeout=10)
    if output:
        return len(output.split("\n"))
    return 0


def _git_last_author(root: str, path: str) -> str:
    """Get the last author who modified a path."""
    return _git_cmd(root, ["log", "-1", "--format=%an", "--", path])
