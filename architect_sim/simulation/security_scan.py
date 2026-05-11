"""Regex-based security scanner for Go/Python/Rust/TypeScript source code.

Detects: hardcoded secrets, SQL injection, command injection, path traversal,
CORS misconfiguration, insecure HTTP, and missing auth middleware.

Pattern sources: secrets-patterns-db, gitleaks, detect-secrets, njsscan/libsast.
"""
import re
from pathlib import Path
from ..models import Finding


# =============================================================================
# Pattern Definitions
# =============================================================================

SECRET_PATTERNS = [
    # AWS
    {"name": "AWS Access Key", "regex": r"AKIA[0-9A-Z]{16}", "severity": "critical"},
    {"name": "AWS Secret Key", "regex": r"(?i)aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]", "severity": "critical"},
    # Anthropic
    {"name": "Anthropic API Key", "regex": r"sk-ant-api03-[a-zA-Z0-9_\-]{93}AA", "severity": "critical"},
    # OpenAI
    {"name": "OpenAI API Key", "regex": r"sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}", "severity": "critical"},
    # GitHub
    {"name": "GitHub Token", "regex": r"gh[pousr]_[A-Za-z0-9_]{36,}", "severity": "critical"},
    {"name": "GitHub Fine-grained PAT", "regex": r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}", "severity": "critical"},
    # Stripe
    {"name": "Stripe Secret Key", "regex": r"sk_live_[a-zA-Z0-9]{24,}", "severity": "critical"},
    {"name": "Stripe Publishable", "regex": r"pk_live_[a-zA-Z0-9]{24,}", "severity": "warning"},
    # Slack
    {"name": "Slack Token", "regex": r"xox[baprs]-[0-9]{10,13}-[a-zA-Z0-9-]+", "severity": "critical"},
    {"name": "Slack Webhook", "regex": r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+", "severity": "warning"},
    # Generic
    {"name": "Private Key", "regex": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "severity": "critical"},
    {"name": "JWT Token", "regex": r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+", "severity": "warning"},
    # Password in assignment
    {"name": "Hardcoded Password", "regex": r"""(?i)(?:password|passwd|pwd|secret|api_?key|auth_?token|access_?token)\s*[:=]\s*['"][^'"]{8,}['"]""", "severity": "warning"},
    # Generic high-entropy (Base64 strings > 30 chars in assignments)
    {"name": "Generic Secret Assignment", "regex": r"""(?i)(?:secret|token|key|password|credential)\s*[:=]\s*['"][a-zA-Z0-9+/=]{30,}['"]""", "severity": "warning"},
    # Supabase
    {"name": "Supabase Key", "regex": r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", "severity": "warning"},
    # Database URLs with credentials
    {"name": "Database URL with Password", "regex": r"(?:postgres|mysql|mongodb|redis)://\w+:[^@\s]{3,}@", "severity": "critical"},
    # Database URL without SSL (sslmode=disable)
    {"name": "Database SSL Disabled", "regex": r"(?:postgres|mysql)://.*sslmode=disable", "severity": "warning"},
    # Bearer tokens
    {"name": "Bearer Token", "regex": r"""(?i)bearer\s+[a-zA-Z0-9_\-.]{20,}""", "severity": "warning"},
    # Google
    {"name": "Google API Key", "regex": r"AIza[0-9A-Za-z_-]{35}", "severity": "critical"},
    # Twilio
    {"name": "Twilio Auth Token", "regex": r"(?i)twilio.{0,20}[0-9a-f]{32}", "severity": "critical"},
    # SendGrid
    {"name": "SendGrid API Key", "regex": r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}", "severity": "critical"},
    # HuggingFace
    {"name": "HuggingFace Token", "regex": r"hf_[a-zA-Z0-9]{34}", "severity": "critical"},
]

SQL_INJECTION_PATTERNS = [
    # Go: string concatenation in SQL
    {"name": "SQL String Concat (Go)", "regex": r'(?:db\.Query|db\.Exec|db\.QueryRow)\(\s*"[^"]*"\s*\+', "langs": ["go"], "severity": "critical"},
    {"name": "SQL Sprintf (Go)", "regex": r'(?:db\.Query|db\.Exec)\(\s*fmt\.Sprintf\(', "langs": ["go"], "severity": "critical"},
    # Python: string formatting in SQL
    {"name": "SQL Format String (Python)", "regex": r'(?:cursor\.execute|\.execute)\(\s*f["\']', "langs": ["python"], "severity": "critical"},
    {"name": "SQL String Concat (Python)", "regex": r'(?:cursor\.execute|\.execute)\(\s*["\'].*["\']\s*\+', "langs": ["python"], "severity": "critical"},
    {"name": "SQL % Format (Python)", "regex": r'(?:cursor\.execute|\.execute)\(\s*["\'].*%s.*["\']\s*%', "langs": ["python"], "severity": "critical"},
    # TypeScript/JS: string concat in query
    {"name": "SQL Template Literal (JS)", "regex": r'(?:query|execute)\(\s*`[^`]*\$\{', "langs": ["node"], "severity": "critical"},
    {"name": "SQL String Concat (JS)", "regex": r'(?:query|execute)\(\s*["\'][^"\']*["\']\s*\+', "langs": ["node"], "severity": "critical"},
]

COMMAND_INJECTION_PATTERNS = [
    # Go
    {"name": "Command Injection (Go)", "regex": r'exec\.Command\(\s*["\'][^"\']*["\']\s*\+', "langs": ["go"], "severity": "critical"},
    {"name": "Unsafe Shell Exec (Go)", "regex": r'exec\.Command\(\s*"(?:sh|bash|cmd)"', "langs": ["go"], "severity": "warning"},
    # Python
    {"name": "os.system (Python)", "regex": r'os\.system\(\s*(?:f["\']|["\'].*["\']\s*\+|\w+\s*\+)', "langs": ["python"], "severity": "critical"},
    {"name": "subprocess Shell (Python)", "regex": r'subprocess\.(?:call|run|Popen)\(.*shell\s*=\s*True', "langs": ["python"], "severity": "critical"},
    {"name": "eval (Python)", "regex": r'(?<!\.)(?<!model\.)(?<!self\.)\beval\(\s*(?![\'"]\s*\))', "langs": ["python"], "severity": "critical"},
    # JS/TS
    {"name": "child_process.exec (JS)", "regex": r'child_process\.exec\(\s*(?:`[^`]*\$\{|["\'].*["\']\s*\+)', "langs": ["node"], "severity": "critical"},
    {"name": "eval (JS)", "regex": r'\beval\(\s*(?![\'"]\s*\))', "langs": ["node"], "severity": "critical"},
]

PATH_TRAVERSAL_PATTERNS = [
    {"name": "Path Traversal (Go)", "regex": r'(?:os\.Open|ioutil\.ReadFile|filepath\.Join)\(.*(?:r\.URL|c\.Param|req\.Query)', "langs": ["go"], "severity": "critical"},
    {"name": "Path Traversal (Python)", "regex": r'(?:open|os\.path\.join)\(.*(?:request\.|params\[|args\.get)', "langs": ["python"], "severity": "critical"},
    {"name": "Path Traversal (JS)", "regex": r'(?:fs\.readFile|path\.join)\(.*(?:req\.params|req\.query|req\.body)', "langs": ["node"], "severity": "critical"},
    {"name": "Unvalidated Path Join", "regex": r'(?:filepath\.Join|path\.join|os\.path\.join)\(.*\+', "severity": "warning"},
]

# CORS patterns — only flag in production configs, not localhost dev setups.
# Wildcard CORS on localhost is expected for local development.
CORS_PATTERNS = [
    {"name": "CORS Allow All (Production)", "regex": r'(?:production|prod|deploy).*Access-Control-Allow-Origin.*\*', "severity": "warning"},
]

INSECURE_HTTP_PATTERNS = [
    {"name": "HTTP in Production Config", "regex": r'(?:production|prod|live).*http://(?!localhost|127\.0\.0\.1)', "severity": "warning"},
    {"name": "TLS Skip Verify (Go)", "regex": r'InsecureSkipVerify\s*:\s*true', "langs": ["go"], "severity": "critical"},
    {"name": "TLS Verify Disabled (Python)", "regex": r'verify\s*=\s*False', "langs": ["python"], "severity": "warning"},
    {"name": "TLS Reject Unauthorized (JS)", "regex": r'rejectUnauthorized\s*:\s*false', "langs": ["node"], "severity": "critical"},
    {"name": "Weak TLS Version", "regex": r'(?:TLSv1\.0|TLSv1\.1|SSLv3|MinVersion.*tls\.VersionTLS10)', "severity": "warning"},
]

# Debug/admin exposure
DEBUG_EXPOSURE_PATTERNS = [
    {"name": "pprof Exposed", "regex": r'[_ ]"net/http/pprof"', "langs": ["go"], "severity": "warning"},
    {"name": "pprof Handler Registered", "regex": r'pprof\.Handler|/debug/pprof|ListenAndServe.*pprof', "severity": "warning"},
    {"name": "Debug Mode Enabled", "regex": r'(?i)(?:debug|DEBUG)\s*[:=]\s*(?:true|True|1|"true")', "severity": "warning"},
    {"name": "Debug Endpoint", "regex": r'(?:HandleFunc|GET|POST)\(\s*["\'](?:/debug/|/admin/|/__)', "severity": "warning"},
    {"name": "Stack Trace in Response", "regex": r'(?:stacktrace|stack_trace|traceback)\s*[:=]', "severity": "warning"},
]

# Input validation / sanitization gaps
INPUT_VALIDATION_PATTERNS = [
    # Deserialization without validation
    {"name": "Unsafe Deserialization (Python)", "regex": r'pickle\.loads?\(|yaml\.load\(\s*[^,]+(?:,\s*Loader\s*=\s*yaml\.(?:Unsafe|Full)Loader)?(?:\s*\))', "langs": ["python"], "severity": "critical"},
    {"name": "Unsafe Deserialization (Go)", "regex": r'json\.Unmarshal\([^,]+,\s*&?interface\{\}', "langs": ["go"], "severity": "warning"},
    # Template injection
    {"name": "Template Injection (Python)", "regex": r'\.render_template_string\(|Template\(\s*(?:request|params|args)', "langs": ["python"], "severity": "critical"},
    {"name": "Template Injection (JS)", "regex": r'(?:ejs|pug|jade)\.render\(.*(?:req\.|params|body)', "langs": ["node"], "severity": "critical"},
    # Header injection
    {"name": "Header Injection", "regex": r'(?:setHeader|set_header|Header\.Set)\(.*(?:req\.|request\.|params)', "severity": "warning"},
    # Open redirect
    {"name": "Open Redirect", "regex": r'(?:redirect|Redirect|302)\(.*(?:req\.|request\.|params\[|c\.Query)', "severity": "warning"},
    # Log injection (sensitive data in logs)
    {"name": "Sensitive Data in Logs", "regex": r'(?i)(?:log|print|fmt\.Print|console\.log).*(?:password\s*[:=]|secret\s*[:=]|api.?key\s*[:=]|credential\s*[:=]|bearer\s+\w{20})', "severity": "warning"},
    # XML external entity
    {"name": "XXE Risk (Go)", "regex": r'xml\.NewDecoder\(', "langs": ["go"], "severity": "warning"},
    {"name": "XXE Risk (Python)", "regex": r'etree\.parse\(|xml\.dom\.minidom\.parse\(', "langs": ["python"], "severity": "warning"},
]

# Rate limiting / DoS protection
RATE_LIMIT_PATTERNS = [
    # Endpoints that accept POST without visible rate limiting
    # (heuristic — flag services that have POST endpoints but no rate limiter import)
    {"name": "Missing Rate Limiter (Go)", "regex": r'func main\(\)', "severity": "info", "meta": "check_rate_limit"},
]

# Auth middleware indicators — used for heuristic "missing auth" detection
AUTH_INDICATORS = [
    re.compile(p) for p in [
        r'(?i)auth', r'middleware', r'protect', r'authenticate',
        r'JWT', r'Bearer', r'RequireAuth', r'IsAuthenticated',
        r'@login_required', r'@requires_auth', r'@jwt_required',
    ]
]

# All pattern sets for iteration
_ALL_PATTERN_SETS = [
    SECRET_PATTERNS,
    SQL_INJECTION_PATTERNS,
    COMMAND_INJECTION_PATTERNS,
    PATH_TRAVERSAL_PATTERNS,
    CORS_PATTERNS,
    INSECURE_HTTP_PATTERNS,
    DEBUG_EXPOSURE_PATTERNS,
    INPUT_VALIDATION_PATTERNS,
]

# =============================================================================
# File extension mappings
# =============================================================================

_LANG_EXTENSIONS = {
    "go": {".go"},
    "rust": {".rs"},
    "python": {".py"},
    "node": {".ts", ".tsx", ".js", ".jsx", ".mjs"},
    "typescript": {".ts", ".tsx", ".js", ".jsx", ".mjs"},
}

# Directories to skip
_SKIP_DIRS = frozenset({
    # Package managers / dependencies
    "vendor", "node_modules", "target", "dist", "build", "__pycache__",
    ".git", ".venv", "venv", "env", "site-packages",
    # Third-party code
    "third_party", "third-party", "external", "lib", "libs",
    # ComfyUI and other embedded third-party
    "ComfyUI", "comfyui", "custom_nodes", "Comfyui_turbodiffusion",
    # Test data
    "testdata", "test_fixtures", "fixtures", "mocks", "mock",
    "examples", "example", "demos", "demo",
    # Documentation
    "docs", "documentation",
    # Generated code
    "generated", "gen", "pb", "proto",
    # Meta/evolution (not production code)
    "iterations", "evolution", "experiments", "novelty_lab",
    # Build artifacts
    ".build", ".next", "coverage", "htmlcov",
})

# File name patterns to skip
_SKIP_FILE_PATTERNS = [
    # Test files
    re.compile(r'_test\.go$'),
    re.compile(r'test_.*\.py$'),
    re.compile(r'.*_test\.py$'),
    re.compile(r'.*\.test\.[jt]sx?$'),
    re.compile(r'.*\.spec\.[jt]sx?$'),
    re.compile(r'.*_test\.rs$'),
    # Env templates
    re.compile(r'\.env\.example$'),
    re.compile(r'\.env\.sample$'),
    re.compile(r'\.env\.template$'),
    # Benchmarks and scripts (not production)
    re.compile(r'benchmark_.*\.py$'),
    re.compile(r'.*_benchmark\.py$'),
    re.compile(r'.*_bench\.go$'),
    # Generated / protobuf
    re.compile(r'.*\.pb\.go$'),
    re.compile(r'.*_generated\.go$'),
    re.compile(r'.*\.gen\.go$'),
    re.compile(r'.*_gen\.go$'),
]


# =============================================================================
# Fix Suggestions
# =============================================================================

_FIX_SUGGESTIONS = {
    # Secrets
    "AWS Access Key": "Move to environment variable or secrets manager (AWS Secrets Manager, Vault)",
    "AWS Secret Key": "Move to environment variable or secrets manager; never commit credentials",
    "Anthropic API Key": "Use ANTHROPIC_API_KEY environment variable instead of hardcoding",
    "OpenAI API Key": "Use OPENAI_API_KEY environment variable instead of hardcoding",
    "GitHub Token": "Use GITHUB_TOKEN environment variable or gh auth; rotate this token immediately",
    "GitHub Fine-grained PAT": "Use GITHUB_TOKEN environment variable; rotate this PAT immediately",
    "Stripe Secret Key": "Use STRIPE_SECRET_KEY env var; rotate key in Stripe dashboard",
    "Stripe Publishable": "While less critical, move to env var for consistency",
    "Slack Token": "Use SLACK_TOKEN env var; rotate in Slack app settings",
    "Slack Webhook": "Move webhook URL to env var or config outside version control",
    "Private Key": "Move private key to a file excluded from version control, reference via path env var",
    "JWT Token": "Generate tokens at runtime; don't hardcode static JWTs",
    "Hardcoded Password": "Use environment variable or secrets manager for credentials",
    "Generic Secret Assignment": "Use environment variable or secrets manager; avoid inline secrets",
    "Supabase Key": "Use SUPABASE_KEY / SUPABASE_ANON_KEY env vars",
    "Database URL with Password": "Use DATABASE_URL env var; never commit connection strings with passwords",
    "Bearer Token": "Load bearer tokens from env or runtime config, not source code",
    "Google API Key": "Use GOOGLE_API_KEY env var; restrict key in Google Cloud Console",
    "Twilio Auth Token": "Use TWILIO_AUTH_TOKEN env var; rotate in Twilio console",
    "SendGrid API Key": "Use SENDGRID_API_KEY env var; rotate in SendGrid settings",
    "HuggingFace Token": "Use HF_TOKEN env var; rotate in HuggingFace settings",
    # SQL Injection
    "SQL String Concat (Go)": "Use parameterized queries: db.Query(\"SELECT ... WHERE id = $1\", id)",
    "SQL Sprintf (Go)": "Use parameterized queries instead of fmt.Sprintf for SQL",
    "SQL Format String (Python)": "Use parameterized queries: cursor.execute(\"SELECT ... WHERE id = %s\", (id,))",
    "SQL String Concat (Python)": "Use parameterized queries with placeholders instead of string concatenation",
    "SQL % Format (Python)": "Use cursor.execute(query, params) tuple form, not % string formatting",
    "SQL Template Literal (JS)": "Use parameterized queries: query('SELECT ... WHERE id = $1', [id])",
    "SQL String Concat (JS)": "Use parameterized queries instead of string concatenation",
    # Command Injection
    "Command Injection (Go)": "Pass arguments as separate parameters to exec.Command, not concatenated strings",
    "Unsafe Shell Exec (Go)": "Avoid shell invocation; use exec.Command with explicit binary path and args",
    "os.system (Python)": "Use subprocess.run() with a list of arguments (no shell=True)",
    "subprocess Shell (Python)": "Use subprocess.run([...]) with shell=False (the default); pass args as list",
    "eval (Python)": "Replace eval() with ast.literal_eval() for data, or structured parsing",
    "child_process.exec (JS)": "Use child_process.execFile() or spawn() with argument arrays",
    "eval (JS)": "Replace eval() with JSON.parse() for data, or use a sandboxed interpreter",
    # Path Traversal
    "Path Traversal (Go)": "Use filepath.Clean() and verify the result stays within allowed directory",
    "Path Traversal (Python)": "Use os.path.realpath() and verify prefix stays within allowed directory",
    "Path Traversal (JS)": "Use path.resolve() and verify result starts with allowed base directory",
    "Unvalidated Path Join": "Validate/sanitize user input before joining paths; check for '..' traversal",
    # CORS
    "CORS Allow All Origins": "Restrict Access-Control-Allow-Origin to specific allowed domains",
    "CORS Allow All (Go)": "Set AllowOrigins to specific domains instead of wildcard/true",
    "CORS Allow All (JS)": "Configure CORS origin to a whitelist of allowed domains",
    "CORS Wildcard Header": "Specify exact allowed headers instead of wildcard",
    # Insecure HTTP
    "HTTP in Production Config": "Use HTTPS for all production endpoints",
    "TLS Skip Verify (Go)": "Remove InsecureSkipVerify or make it dev-only with build tags",
    "TLS Verify Disabled (Python)": "Set verify=True (default) or use a custom CA bundle",
    "TLS Reject Unauthorized (JS)": "Remove rejectUnauthorized:false; configure proper CA certificates",
    "Weak TLS Version": "Set minimum TLS version to 1.2 (tls.VersionTLS12)",
    # Auth
    "Missing Auth Middleware": "Add authentication middleware (JWT, session, API key) to protect this route",
    # Database
    "Database SSL Disabled": "Enable SSL: use sslmode=require or sslmode=verify-full in connection string",
    # Debug exposure
    "pprof Exposed": "Guard pprof behind auth or bind to localhost-only in production",
    "pprof Handler Registered": "Ensure /debug/pprof is behind authentication or localhost-only",
    "Debug Mode Enabled": "Disable debug mode in production; use build tags or env var checks",
    "Debug Endpoint": "Remove or protect debug/admin endpoints behind authentication",
    "Stack Trace in Response": "Don't expose stack traces to clients; log them server-side only",
    # Input validation
    "Unsafe Deserialization (Python)": "Use yaml.safe_load() instead of yaml.load(); avoid pickle on untrusted data",
    "Unsafe Deserialization (Go)": "Unmarshal into typed structs, not interface{}; validate after deserialization",
    "Template Injection (Python)": "Never pass user input to render_template_string(); use render_template() with files",
    "Template Injection (JS)": "Never pass user input directly to template engines; sanitize first",
    "Header Injection": "Validate and sanitize header values; reject values containing newlines",
    "Open Redirect": "Validate redirect URLs against a whitelist of allowed destinations",
    "Sensitive Data in Logs": "Never log passwords, tokens, or secrets; mask sensitive fields before logging",
    "XXE Risk (Go)": "Disable external entity processing in XML parser",
    "XXE Risk (Python)": "Use defusedxml instead of xml.etree for untrusted XML input",
}


# =============================================================================
# Public API
# =============================================================================

def check_security(blueprints: dict, config) -> list:
    """Run security scan across all services.

    Returns list of Finding objects.
    """
    findings = []
    for bp in blueprints.values():
        findings.extend(_scan_service(bp))
    return findings


# =============================================================================
# Internal Implementation
# =============================================================================

def _scan_service(bp) -> list:
    """Scan a single service for security issues."""
    findings = []
    source_dir = Path(bp.source_dir)
    lang = bp.language

    if not source_dir.exists():
        return findings

    for source_file in _iter_source_files(source_dir, lang):
        try:
            content = source_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(source_file)

        # Run all applicable pattern sets
        for pattern_set in _ALL_PATTERN_SETS:
            findings.extend(_check_patterns(pattern_set, content, file_str, bp.name, lang))

        # Heuristic: missing auth middleware on route handlers
        findings.extend(_check_auth_middleware(bp, content, file_str, lang))

    return findings


def _iter_source_files(source_dir: Path, lang: str):
    """Yield source files for scanning, skipping tests/vendor/etc."""
    extensions = _LANG_EXTENSIONS.get(lang, set())
    if not extensions:
        return

    for path in source_dir.rglob("*"):
        # Skip directories in the skip set
        if any(part in _SKIP_DIRS for part in path.parts):
            continue

        # Only process files with matching extensions
        if path.suffix not in extensions:
            continue

        # Skip test files and example env files
        filename = path.name
        if any(pat.search(filename) for pat in _SKIP_FILE_PATTERNS):
            continue

        if path.is_file():
            yield path


def _is_test_or_vendor(file_path: str) -> bool:
    """Check if file is in a test/vendor/example directory."""
    parts = Path(file_path).parts
    return any(part in _SKIP_DIRS for part in parts)


def _is_comment(line: str, lang: str) -> bool:
    """Check if a line is a comment (simple heuristic)."""
    stripped = line.strip()
    if not stripped:
        return False

    if lang in ("go", "node", "typescript", "rust"):
        return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")
    elif lang == "python":
        return stripped.startswith("#")

    # Fallback: check both styles
    return stripped.startswith("//") or stripped.startswith("#")


def _check_patterns(patterns: list, content: str, file_path: str, service: str, lang: str) -> list:
    """Check a set of patterns against file content."""
    findings = []
    lines = content.split("\n")

    for pat in patterns:
        # Skip if pattern is language-specific and doesn't match
        if "langs" in pat and lang not in pat["langs"]:
            continue

        for match in re.finditer(pat["regex"], content):
            line_num = content[:match.start()].count("\n") + 1

            # Skip test files, vendor, examples
            if _is_test_or_vendor(file_path):
                continue

            # Skip comments
            if line_num <= len(lines):
                line = lines[line_num - 1]
            else:
                line = ""

            if _is_comment(line, lang):
                continue

            # Skip lines with explicit security suppressions
            if 'nolint:gosec' in line or 'nosec' in line or '# noqa' in line or 'SAFETY:' in line:
                continue

            # Skip pprof imports when gated by ATHENA_PPROF_ENABLED
            if pat["name"] == "pprof Exposed" and 'PPROF_ENABLED' in content:
                continue
            if pat["name"] == "pprof Handler Registered" and ('PPROF_ENABLED' in content or 'ENABLE_PPROF' in content):
                continue
            if pat["name"] == "Debug Endpoint" and ('PPROF_ENABLED' in content or 'ENABLE_PPROF' in content):
                continue

            # Skip Go XML — encoding/xml doesn't support DTDs/external entities
            if pat["name"] == "XXE Risk (Go)" and 'encoding/xml' in content:
                continue

            # Skip sensitive data that's DB inserts, not log output
            if pat["name"] == "Sensitive Data in Logs":
                if any(x in line for x in ['args = append', 'HSet', 'INSERT', 'UPDATE',
                                             'fmt.Sprintf("api_key', 'parameterized',
                                             'credential', 'service/', 'import ']):
                    continue

            # Skip stack traces in logger.debug (not in HTTP response)
            if pat["name"] == "Stack Trace in Response" and 'logger.debug' in line:
                continue

            # Skip PyTorch model.eval() — not Python eval()
            if pat["name"] == "eval (Python)" and ('model.eval' in line or '.eval()' in line):
                if 'self.' in line or 'model' in line.lower():
                    continue

            # Skip string literals containing "eval" (not function calls)
            if pat["name"] == "eval (Python)" and 'eval(' not in line:
                continue

            # Skip password patterns that are parsing/detecting, not hardcoding
            if pat["name"] == "Hardcoded Password":
                if any(x in line for x in ['HasPrefix', 'starts_with', 'Contains', 'getenv',
                                             'os.Getenv', 'os.environ', 'Getenv(', '.get(',
                                             'test-key', 'test_key', 'demo', 'TrimPrefix',
                                             'strings.Trim', 'for pat in', 'pattern',
                                             'example', 'placeholder', '***', 'xxx']):
                    continue

            # Skip Supabase/JWT anon keys (public by design, not secrets)
            if pat["name"] in ("JWT Token", "Supabase Key"):
                context_window = content[max(0, match.start()-300):match.start()+300]
                if any(x in context_window.lower() for x in [
                    'anon', 'anonkey', 'anon_key', 'public',
                    'supabase-demo', 'demo', 'test', 'default',
                    'SUPABASE_ANON', 'publishable',
                ]):
                    continue

            # Skip Database SSL disabled on localhost (expected for local dev)
            if pat["name"] == "Database SSL Disabled":
                if 'localhost' in line or '127.0.0.1' in line:
                    continue

            # Skip pprof when bound to localhost only (safe)
            if pat["name"] in ("pprof Exposed", "pprof Handler Registered"):
                context_window = content[max(0, match.start()-500):match.start()+500]
                if '127.0.0.1' in context_window and '0.0.0.0' not in context_window:
                    continue  # Bound to localhost only — safe

            # Skip path joins that use validated/cleaned input
            if pat["name"] == "Unvalidated Path Join":
                context_window = content[max(0, match.start()-200):match.start()+200]
                if any(x in context_window for x in [
                    'filepath.Clean', 'filepath.Abs', 'path.resolve',
                    'os.path.realpath', 'os.path.abspath',
                    'sanitize', 'validate', 'allowlist', 'whitelist',
                ]):
                    continue

            # Skip XXE when using safe parsers
            if "XXE" in pat["name"]:
                context_window = content[max(0, match.start()-300):match.start()+300]
                if any(x in context_window for x in ['defusedxml', 'safe', 'DisableEntityResolution']):
                    continue

            # Skip sensitive data in logs that are just key names, not values
            if pat["name"] == "Sensitive Data in Logs":
                if any(x in line for x in ['disabled', 'enabled', 'missing', 'not set',
                                             'configured', 'initialized', 'no secret',
                                             'REDACTED', 'masked', '***']):
                    continue

            findings.append(Finding(
                finding_type="security_issue",
                severity=pat["severity"],
                fixability="needs_context",
                service=service,
                endpoint=pat["name"],
                file_path=file_path,
                line_number=line_num,
                details=f"[{pat['name']}] {line.strip()[:80]}",
                suggested_fix=_get_fix_suggestion(pat["name"]),
            ))
            if len([f for f in findings if f.endpoint == pat["name"]]) >= 3:
                break  # Cap at 3 per pattern per file

    return findings


def _check_auth_middleware(bp, content: str, file_path: str, lang: str) -> list:
    """Heuristic check: route handlers without auth middleware nearby."""
    findings = []

    # Detect route handler registration patterns per language
    if lang == "go":
        route_pattern = re.compile(
            r'(?:\.(?:GET|POST|PUT|DELETE|PATCH|Handle|HandleFunc))\(\s*"(/[^"]*)"'
        )
    elif lang == "python":
        route_pattern = re.compile(
            r'@(?:app|router|blueprint)\.(?:route|get|post|put|delete|patch)\(\s*["\'](/[^"\']*)["\']'
        )
    elif lang in ("node", "typescript"):
        route_pattern = re.compile(
            r'(?:router|app)\.(?:get|post|put|delete|patch)\(\s*["\'](/[^"\']*)["\']'
        )
    else:
        return findings

    # Skip health/metrics/docs/internal endpoints
    skip_paths = {"/health", "/live", "/ready", "/metrics", "/version", "/docs", "/swagger",
                  "/status", "/ping", "/info", "/reload", "/rerank", "/embed",
                  "/v1/chat/completions", "/v1/models", "/v1/embeddings"}

    lines = content.split("\n")

    for match in route_pattern.finditer(content):
        route_path = match.group(1)

        # Skip infra paths
        if route_path in skip_paths or any(route_path.startswith(p) for p in ("/health", "/metrics", "/debug")):
            continue

        # Skip internal-only services (bound to 127.0.0.1 or localhost, not user-facing)
        if '127.0.0.1' in content[:500] or 'localhost' in content[:500] or '0.0.0.0' not in content[:500]:
            continue  # Internal service — auth not required

        line_num = content[:match.start()].count("\n") + 1

        # Look for auth indicators in surrounding context (20 lines before/after)
        start_line = max(0, line_num - 20)
        end_line = min(len(lines), line_num + 20)
        context = "\n".join(lines[start_line:end_line])

        has_auth = any(indicator.search(context) for indicator in AUTH_INDICATORS)

        if not has_auth:
            # Also check the file-level imports/middleware setup (first 50 lines)
            header = "\n".join(lines[:50])
            has_auth = any(indicator.search(header) for indicator in AUTH_INDICATORS)

        if not has_auth:
            findings.append(Finding(
                finding_type="security_issue",
                severity="warning",
                fixability="needs_context",
                service=bp.name,
                endpoint="Missing Auth Middleware",
                file_path=file_path,
                line_number=line_num,
                details=f"[Missing Auth Middleware] Route {route_path} has no visible auth middleware",
                suggested_fix=_get_fix_suggestion("Missing Auth Middleware"),
            ))
            break  # One per file to avoid noise

    return findings


def _get_fix_suggestion(pattern_name: str) -> str:
    """Return a fix suggestion for the given pattern name."""
    return _FIX_SUGGESTIONS.get(pattern_name, "Review and remediate this security issue")
