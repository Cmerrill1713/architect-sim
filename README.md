# architect-sim

Static analysis simulator for microservice architectures. Extracts service blueprints from source code, simulates every request path and failure mode, grades the system, and auto-fixes issues -- all without running a single service.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Analyze any project
architect-sim --root /path/to/project

# JSON output
architect-sim --root /path/to/project --format json

# Specify services directory
architect-sim --root /path/to/project --services-dir src/services

# With fix iterations
architect-sim --root /path/to/project --max-iterations 5000

# Analyze only (no fixes)
architect-sim --root /path/to/project --analyze-only
```

## What It Finds

- **Phantom calls**: Service A calls an endpoint on Service B that doesn't exist
- **Contract mismatches**: Caller uses GET but endpoint expects POST
- **Dead code**: Calls to ports/services that don't exist
- **Circular dependencies**: Services that depend on each other
- **Orphan endpoints**: Defined routes that nothing calls
- **Documentation drift**: Docs referencing services/ports not in code
- **Missing retries**: Inter-service calls without retry wrappers

## How It Works

1. **Extract** -- Parse Go/Rust/Python source to find HTTP routes and inter-service calls
2. **Simulate** -- Verify every call lands on a real endpoint, check contracts
3. **Grade** -- Score across 6 dimensions (completeness, contracts, resilience, deps, coverage, security)
4. **Fix** -- Apply deterministic IF/THEN rules to resolve issues
5. **Re-grade** -- Verify fixes improved the score

## Supported Languages

- Go (gin, gorilla/mux, net/http)
- Rust (axum)
- Python (FastAPI, Flask)

## Configuration

The tool auto-detects project structure. It looks for:
- `ports.yaml` or `docker-compose.yml` for port assignments
- Directories containing `.go`, `.rs`, or `.py` files as services
- `service_contracts.yaml` or similar for declared dependencies

You can override auto-detection with CLI flags.
