"""Auto-apply engine -- close the loop from analysis to committed fixes.

Takes accepted fixes from the fast_engine, generates real file patches,
applies them to disk, verifies they compile, and optionally commits.

Safety:
- Dry run by default (--apply flag required)
- Build gate: changes that don't compile are automatically reverted
- Per-file rollback: if one patch fails, only that patch is reverted
- Summary before commit unless --auto-commit
"""
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .patch_generator import generate_patches, generate_diff, Patch
from ..verification.build_check import verify_build, BuildResult


@dataclass
class ApplyResult:
    patches_generated: int = 0
    patches_applied: int = 0
    patches_failed: int = 0
    build_passed: bool = True
    files_changed: list = field(default_factory=list)
    build_output: str = ""
    reverted: list = field(default_factory=list)
    committed: bool = False
    commit_hash: str = ""
    dry_run: bool = True
    errors: list = field(default_factory=list)


def auto_apply(fixes: list, blueprints: dict, config,
               dry_run: bool = True,
               auto_commit: bool = False,
               verify_build_flag: bool = True,
               min_severity: str = "warning") -> ApplyResult:
    """Apply fixes to actual source files.

    Args:
        fixes: Accepted Fix objects from fast_engine
        blueprints: Service blueprints for file resolution
        config: Config for root path
        dry_run: If True, only show what would change
        auto_commit: If True, commit changes automatically
        verify_build_flag: If True, run build checks after applying
        min_severity: Only apply fixes at or above this severity

    Steps:
        1. Filter fixes by severity and action type
        2. Generate patches from fixes (patch_generator)
        3. If dry_run: show patches and return
        4. Apply each patch to disk
        5. Run build verification for affected services
        6. If build fails for a service:
           a. Revert all patches for that service
           b. Log the failure
        7. If auto_commit and changes remain:
           a. git add changed files
           b. git commit with summary
    """
    result = ApplyResult(dry_run=dry_run)

    # Filter fixes: only accepted, only code-modifying actions, severity gate
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    min_rank = severity_rank.get(min_severity, 1)

    applicable = [
        f for f in fixes
        if f.accepted
        and f.action in ("add_endpoint", "fix_method", "remove_call")
        and f.finding
        and severity_rank.get(f.finding.severity, 9) <= min_rank
    ]

    if not applicable:
        return result

    # Generate patches
    root = str(config.root)
    patches = generate_patches(applicable, blueprints, root)
    result.patches_generated = len(patches)

    if not patches:
        return result

    # Dry run: just count, don't touch disk
    if dry_run:
        result.files_changed = [p.file_path for p in patches]
        return result

    # Apply patches one by one with rollback capability
    applied_patches = []
    for patch in patches:
        success, error = _apply_patch_safe(patch)
        if success:
            applied_patches.append(patch)
            result.patches_applied += 1
            if patch.file_path not in result.files_changed:
                result.files_changed.append(patch.file_path)
        else:
            result.patches_failed += 1
            result.errors.append(f"{patch.file_path}: {error}")

    # Build verification
    if verify_build_flag and applied_patches:
        # Group patches by service for targeted verification
        service_patches = _group_by_service(applied_patches, blueprints)

        for service_name, svc_patches in service_patches.items():
            bp = blueprints.get(service_name)
            if not bp:
                continue

            build_result = verify_build(bp.source_dir, bp.language)
            result.build_output += f"[{service_name}] {build_result.command}: "

            if build_result.success:
                result.build_output += "PASS\n"
            else:
                result.build_output += f"FAIL\n{build_result.output}\n"
                result.build_passed = False

                # Revert all patches for this service
                for patch in svc_patches:
                    _revert_patch(patch)
                    result.reverted.append(patch)
                    result.patches_applied -= 1
                    if patch.file_path in result.files_changed:
                        result.files_changed.remove(patch.file_path)

                result.errors.append(
                    f"Build failed for {service_name}, reverted {len(svc_patches)} patches"
                )

    # If nothing survived verification, we're done
    if not result.files_changed:
        return result

    # All remaining builds passed
    if not result.reverted:
        result.build_passed = True

    # Commit
    if auto_commit and result.files_changed:
        summary = _build_commit_message(applied_patches, result)
        success, commit_hash = _commit_changes(
            result.files_changed, root, summary
        )
        result.committed = success
        result.commit_hash = commit_hash
        if not success:
            result.errors.append(f"Commit failed: {commit_hash}")

    return result


def _apply_patch_safe(patch: Patch) -> tuple:
    """Apply a single patch with rollback capability.

    Saves original content before modifying, restores on failure.
    Returns (success: bool, error: str).
    """
    path = Path(patch.file_path)

    if not path.exists():
        return False, f"File not found: {patch.file_path}"

    try:
        # Verify the file still matches what we expect
        current = path.read_text(errors="replace")
        if current != patch.original_content:
            return False, "File has been modified since patch was generated"

        path.write_text(patch.patched_content)
        return True, ""
    except (OSError, IOError) as e:
        # Attempt to restore original on write failure
        try:
            path.write_text(patch.original_content)
        except (OSError, IOError):
            pass
        return False, str(e)


def _revert_patch(patch: Patch):
    """Restore original file content from patch.original_content."""
    path = Path(patch.file_path)
    try:
        path.write_text(patch.original_content)
    except (OSError, IOError):
        pass  # Best effort -- file may have been removed


def _group_by_service(patches: list, blueprints: dict) -> dict:
    """Group patches by their owning service.

    Returns {service_name: [patches]}.
    """
    groups = {}
    for patch in patches:
        matched = None
        for name, bp in blueprints.items():
            if bp.source_dir and patch.file_path.startswith(bp.source_dir):
                matched = name
                break
        if matched is None:
            matched = "__unknown__"
        groups.setdefault(matched, []).append(patch)
    return groups


def _build_commit_message(patches: list, result: ApplyResult) -> str:
    """Build a descriptive commit message from applied patches."""
    lines = [f"architect-sim: auto-apply {result.patches_applied} fixes"]
    lines.append("")

    for patch in patches:
        if patch not in result.reverted:
            lines.append(f"- {patch.description}")

    if result.reverted:
        lines.append("")
        lines.append(f"Reverted {len(result.reverted)} patches (build failure)")

    return "\n".join(lines)


def _commit_changes(files: list, root: str, summary: str) -> tuple:
    """Git commit the changed files.

    Returns (success: bool, commit_hash_or_error: str).
    """
    try:
        # Stage only the files we changed
        add_result = subprocess.run(
            ["git", "add"] + files,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if add_result.returncode != 0:
            return False, add_result.stderr.strip()

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", summary],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit_result.returncode != 0:
            return False, commit_result.stderr.strip()

        # Get commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else ""
        return True, commit_hash
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def format_apply_report(result: ApplyResult) -> str:
    """Format the apply results as markdown."""
    lines = []
    lines.append("\n# Auto-Apply Report\n")

    if result.dry_run:
        lines.append("**Mode: DRY RUN** (use `--apply` to modify files)\n")
    else:
        lines.append("**Mode: LIVE APPLY**\n")

    # Summary table
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Patches generated | {result.patches_generated} |")
    lines.append(f"| Patches applied | {result.patches_applied} |")
    lines.append(f"| Patches failed | {result.patches_failed} |")
    lines.append(f"| Patches reverted | {len(result.reverted)} |")
    lines.append(f"| Build passed | {'YES' if result.build_passed else 'NO'} |")
    if result.committed:
        lines.append(f"| Committed | `{result.commit_hash}` |")
    lines.append("")

    # Files changed
    if result.files_changed:
        lines.append("## Files Changed\n")
        for f in result.files_changed:
            lines.append(f"- `{f}`")
        lines.append("")

    # Reverted patches
    if result.reverted:
        lines.append("## Reverted (build failure)\n")
        for patch in result.reverted:
            lines.append(f"- `{patch.file_path}`: {patch.description}")
        lines.append("")

    # Build output
    if result.build_output:
        lines.append("## Build Verification\n")
        lines.append(f"```\n{result.build_output.rstrip()}\n```\n")

    # Errors
    if result.errors:
        lines.append("## Errors\n")
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)
