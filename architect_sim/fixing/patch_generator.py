"""Generate real source code patches from accepted fixes.

Takes in-memory fixes and produces:
1. Unified diff patches (git apply compatible)
2. Direct file edits
3. A summary of what was changed and why
"""
import os
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Patch:
    file_path: str
    original_content: str
    patched_content: str
    description: str
    rule_id: str
    diff: str = ""  # unified diff format


def generate_patches(fixes: list, blueprints: dict, root: str) -> list:
    """Generate file patches from accepted fixes.

    Only generates patches for fixes that modify actual source files:
    - add_endpoint: Adds a handler function + route registration
    - fix_method: Changes the HTTP method in a caller
    - remove_call: Comments out or removes the dead call

    Returns list of Patch objects.
    """
    patches = []

    for fix in fixes:
        if fix.action == "add_endpoint":
            patch = _patch_add_endpoint(fix, blueprints, root)
            if patch:
                patches.append(patch)
        elif fix.action == "fix_method":
            patch = _patch_fix_method(fix, blueprints, root)
            if patch:
                patches.append(patch)
        elif fix.action == "remove_call":
            patch = _patch_remove_call(fix, blueprints, root)
            if patch:
                patches.append(patch)

    return patches


def apply_patches(patches: list, dry_run: bool = True) -> list:
    """Apply patches to actual files.

    Args:
        patches: List of Patch objects
        dry_run: If True, only report what would change without writing

    Returns:
        List of (patch, success, error) tuples
    """
    results = []
    for patch in patches:
        if dry_run:
            results.append((patch, True, "dry run"))
            continue
        try:
            path = Path(patch.file_path)
            if not path.exists():
                results.append((patch, False, f"File not found: {patch.file_path}"))
                continue
            path.write_text(patch.patched_content)
            results.append((patch, True, None))
        except (OSError, IOError) as e:
            results.append((patch, False, str(e)))
    return results


def generate_diff(original: str, patched: str, file_path: str) -> str:
    """Generate unified diff between original and patched content."""
    import difflib
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines, patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


def format_patch_report(patches: list) -> str:
    """Format patches as a readable report."""
    if not patches:
        return "No patches generated.\n"
    lines = [f"## Generated Patches ({len(patches)})\n"]
    for i, p in enumerate(patches, 1):
        lines.append(f"### Patch {i}: {p.description}")
        lines.append(f"File: `{p.file_path}`")
        lines.append(f"Rule: {p.rule_id}")
        if p.diff:
            lines.append(f"```diff\n{p.diff}\n```")
        lines.append("")
    return "\n".join(lines)


def _patch_add_endpoint(fix, blueprints, root):
    """Generate a stub endpoint handler for Go services."""
    mutations = fix.mutations
    target = mutations.get("service", "")
    method = mutations.get("method", "POST")
    path = mutations.get("path", "/")

    if target not in blueprints:
        return None

    bp = blueprints[target]
    if bp.language != "go":
        return None  # Only Go patches for now

    # Find the main.go or router file
    main_file = _find_main_file(bp.source_dir)
    if not main_file:
        return None

    try:
        original = Path(main_file).read_text(errors="replace")
    except (OSError, IOError):
        return None

    # Generate handler name from path
    handler_name = _path_to_handler(path, method)

    # Generate stub handler
    stub = f'''
// {handler_name} - auto-generated stub
// TODO: Implement - called by other services expecting {method} {path}
func {handler_name}(w http.ResponseWriter, r *http.Request) {{
\tw.Header().Set("Content-Type", "application/json")
\tw.WriteHeader(http.StatusNotImplemented)
\tw.Write([]byte(`{{"error":"not implemented","endpoint":"{method} {path}"}}` + "\\n"))
}}
'''

    # Append stub to end of file
    patched = original.rstrip() + "\n" + stub + "\n"

    diff = generate_diff(original, patched, main_file)

    return Patch(
        file_path=main_file,
        original_content=original,
        patched_content=patched,
        description=f"Add stub handler {handler_name} for {method} {path} on {target}",
        rule_id=fix.mutations.get("rule_id", ""),
        diff=diff,
    )


def _patch_fix_method(fix, blueprints, root):
    """Fix HTTP method mismatch in caller code."""
    mutations = fix.mutations
    old_method = mutations.get("old_method", "")
    new_method = mutations.get("new_method", "")
    path = mutations.get("path", "")

    if not old_method or not new_method or not fix.finding:
        return None

    file_path = fix.finding.file_path
    if not file_path or not Path(file_path).exists():
        return None

    try:
        original = Path(file_path).read_text(errors="replace")
    except (OSError, IOError):
        return None

    # Find and replace the method near the path reference
    # Look for patterns like: .Get(url) or http.Get(url) where url contains the path
    # This is a targeted replacement near the line number
    lines = original.split("\n")
    line_idx = fix.finding.line_number - 1

    if 0 <= line_idx < len(lines):
        line = lines[line_idx]
        # Try common patterns
        old_func = f".{old_method}(" if old_method in ("Get", "Post", "Put", "Delete") else f".{old_method.capitalize()}("
        new_func = f".{new_method}(" if new_method in ("Get", "Post", "Put", "Delete") else f".{new_method.capitalize()}("

        if old_func in line:
            lines[line_idx] = line.replace(old_func, new_func, 1)
            patched = "\n".join(lines)
            diff = generate_diff(original, patched, file_path)
            return Patch(
                file_path=file_path,
                original_content=original,
                patched_content=patched,
                description=f"Change {old_method} to {new_method} for {path}",
                rule_id=fix.mutations.get("rule_id", ""),
                diff=diff,
            )

    return None


def _patch_remove_call(fix, blueprints, root):
    """Comment out a dead call in source code."""
    if not fix.finding:
        return None

    file_path = fix.finding.file_path
    if not file_path or not Path(file_path).exists():
        return None

    try:
        original = Path(file_path).read_text(errors="replace")
    except (OSError, IOError):
        return None

    lines = original.split("\n")
    line_idx = fix.finding.line_number - 1

    if 0 <= line_idx < len(lines):
        line = lines[line_idx]
        # Comment out the line with explanation
        indent = len(line) - len(line.lstrip())
        comment_prefix = "//" if _is_go_file(file_path) else "#"
        lines[line_idx] = " " * indent + f"{comment_prefix} DEAD CODE (architect-sim): {line.strip()}"

        patched = "\n".join(lines)
        diff = generate_diff(original, patched, file_path)
        return Patch(
            file_path=file_path,
            original_content=original,
            patched_content=patched,
            description=f"Comment out dead call at {file_path}:{fix.finding.line_number}",
            rule_id=fix.mutations.get("rule_id", ""),
            diff=diff,
        )

    return None


def _find_main_file(source_dir: str) -> str:
    """Find the main source file in a service directory."""
    source_path = Path(source_dir)
    # Prefer main.go, then server.go, then app.go
    for name in ["main.go", "server.go", "app.go", "handlers.go"]:
        candidate = source_path / name
        if candidate.exists():
            return str(candidate)
    # Fall back to any .go file
    go_files = list(source_path.glob("*.go"))
    if go_files:
        return str(go_files[0])
    return None


def _path_to_handler(path: str, method: str) -> str:
    """Convert a URL path to a Go handler function name."""
    # /incidents/service/:param -> handleGetIncidentsService
    clean = path.split("?")[0]  # strip query params
    clean = re.sub(r'[/:{}%]', ' ', clean)  # replace special chars
    parts = clean.split()
    camel = "".join(p.capitalize() for p in parts if p)
    return f"handle{method.capitalize()}{camel}"


def _is_go_file(path: str) -> bool:
    return path.endswith(".go")
