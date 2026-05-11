"""Core data structures for architect-sim."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Endpoint:
    service_name: str        # e.g., "user-service"
    port: int                # e.g., 8080
    method: str              # GET, POST, etc.
    path: str                # e.g., "/api/users"
    file_path: str           # absolute path to source
    line_number: int
    handler_name: str = ""
    request_schema: dict = field(default_factory=dict)   # {field: type}
    response_schema: dict = field(default_factory=dict)
    framework: str = ""      # "gin", "mux", "axum", "fastapi", "flask"


@dataclass
class Call:
    caller_service: str
    callee_service: str      # resolved from port
    callee_port: int
    method: str              # HTTP method
    path: str                # target path
    file_path: str
    line_number: int
    resolution_method: str   # "discovery", "ports", "hardcoded", "serviceURLs"
    request_fields: list = field(default_factory=list)
    retry_config: Optional[dict] = None


@dataclass
class Schema:
    name: str                # struct/class name
    fields: dict             # {field_name: field_type}
    json_tags: dict          # {field_name: json_tag}
    file_path: str = ""
    line_number: int = 0


@dataclass
class ServiceBlueprint:
    name: str
    port: int
    language: str            # go, rust, python
    endpoints: list = field(default_factory=list)       # list[Endpoint]
    outbound_calls: list = field(default_factory=list)  # list[Call]
    schemas: dict = field(default_factory=dict)          # {name: Schema}
    retry_configs: dict = field(default_factory=dict)
    fail_open_endpoints: set = field(default_factory=set)
    source_dir: str = ""


@dataclass
class Finding:
    finding_type: str        # "phantom_call", "orphan_endpoint", "field_mismatch", "missing_endpoint", "undeclared_dep", "circular_dep", "timeout_issue", "dead_code", "port_conflict"
    severity: str            # "critical", "warning", "info"
    fixability: str          # "auto", "needs_context", "needs_user"
    service: str             # primary service affected
    endpoint: str = ""       # relevant endpoint
    file_path: str = ""
    line_number: int = 0
    details: str = ""
    suggested_fix: str = ""
    impact: int = 0          # number of affected components


@dataclass
class FlowTrace:
    name: str                # e.g., "payment_flow"
    steps: list = field(default_factory=list)  # list of dicts: {service, endpoint, status, details}
    success: bool = True


@dataclass
class GradeReport:
    completeness: float = 0.0
    contract_fidelity: float = 0.0
    resilience: float = 0.0
    dependency_hygiene: float = 0.0
    coverage: float = 0.0
    security: float = 0.0
    overall: float = 0.0
    grade: str = "F"         # A, B, C, D, F
    details: dict = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    finding: Optional[Finding] = None
    fix_applied: str = ""
    files_changed: list = field(default_factory=list)
    score_before: dict = field(default_factory=dict)
    score_after: dict = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""         # "improved", "regressed", "reverted", "user_decision"
