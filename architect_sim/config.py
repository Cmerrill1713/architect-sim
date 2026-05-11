"""Configuration loader with auto-detection for any microservice project."""
import json
import os
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: str) -> dict:
    """Load YAML file. Falls back to basic parsing if pyyaml not available."""
    if yaml:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    raise ImportError("pyyaml required: pip install pyyaml")


class Config:
    def __init__(self, root: str, services_dir: str = None, config_file: str = None):
        self.root = Path(root)
        self.ports = {}           # {service_name: port_int}
        self.port_to_service = {} # {port_int: service_name}
        self.contracts = {}       # {service_name: contract_dict}
        self.services_dir = Path(services_dir) if services_dir else None
        self.config_file = config_file
        self._load()

    def _load(self):
        # Load port config from specified file or auto-detect
        if self.config_file:
            self._load_port_config(Path(self.config_file))
        else:
            self._auto_detect_port_config()

        # Load service contracts if available
        self._auto_detect_contracts()

        # Auto-detect services directory if not specified
        if self.services_dir is None:
            self.services_dir = self._auto_detect_services_dir()

    def _load_port_config(self, path: Path):
        """Load port configuration from a specific file."""
        if not path.exists():
            return

        if path.name == "docker-compose.yml" or path.name == "docker-compose.yaml":
            self._load_docker_compose(path)
        elif path.suffix in (".yaml", ".yml"):
            self._load_yaml_ports(path)
        elif path.suffix == ".json":
            self._load_json_ports(path)

    def _load_yaml_ports(self, path: Path):
        """Load ports from a YAML file. Supports multiple schemas."""
        data = load_yaml(str(path))

        # Schema 1: ports.yaml with {ports: {name: {port: N}}}
        if "ports" in data and isinstance(data["ports"], dict):
            for name, info in data["ports"].items():
                if isinstance(info, dict):
                    port = info.get("port")
                    if port:
                        self.ports[name] = int(port)
                        self.port_to_service[int(port)] = name
                elif isinstance(info, int):
                    self.ports[name] = info
                    self.port_to_service[info] = name
            return

        # Schema 2: flat {service_name: port_number}
        for name, val in data.items():
            if isinstance(val, int) and 1024 < val < 65535:
                self.ports[name] = val
                self.port_to_service[val] = name
            elif isinstance(val, dict) and "port" in val:
                port = int(val["port"])
                self.ports[name] = port
                self.port_to_service[port] = name

    def _load_docker_compose(self, path: Path):
        """Extract port mappings from docker-compose.yml."""
        data = load_yaml(str(path))
        services = data.get("services", {})
        for name, svc_config in services.items():
            ports = svc_config.get("ports", [])
            for port_mapping in ports:
                port_str = str(port_mapping)
                # Parse "host:container" or just "port"
                match = re.match(r'(\d+):(\d+)', port_str)
                if match:
                    host_port = int(match.group(1))
                    self.ports[name] = host_port
                    self.port_to_service[host_port] = name
                elif port_str.isdigit():
                    port = int(port_str)
                    self.ports[name] = port
                    self.port_to_service[port] = name

    def _load_json_ports(self, path: Path):
        """Load ports from a JSON file (e.g., service-manifest.json)."""
        with open(path) as f:
            data = json.load(f)

        # Handle service-manifest format: {services: [{name, port}, ...]}
        if "services" in data and isinstance(data["services"], list):
            for svc in data["services"]:
                name = svc.get("name", "")
                port = svc.get("port", 0)
                if name and port:
                    self.ports[name] = int(port)
                    self.port_to_service[int(port)] = name
            return

        # Handle flat dict format
        for name, val in data.items():
            if isinstance(val, int) and 1024 < val < 65535:
                self.ports[name] = val
                self.port_to_service[val] = name

    def _auto_detect_port_config(self):
        """Auto-detect port configuration files in the project."""
        candidates = [
            self.root / "config" / "ports.yaml",
            self.root / "config" / "ports.yml",
            self.root / "ports.yaml",
            self.root / "ports.yml",
            self.root / "docker-compose.yml",
            self.root / "docker-compose.yaml",
            self.root / "service-manifest.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                self._load_port_config(candidate)
                if self.ports:
                    return

        # Scan for any YAML with port-like entries in config/ or project root
        for search_dir in [self.root / "config", self.root]:
            if not search_dir.is_dir():
                continue
            for f in sorted(search_dir.iterdir()):
                if f.suffix in (".yaml", ".yml") and f.is_file():
                    try:
                        self._load_yaml_ports(f)
                        if self.ports:
                            return
                    except Exception:
                        continue

    def _auto_detect_contracts(self):
        """Auto-detect service contracts/dependency declarations."""
        candidates = [
            "service_contracts.yaml",
            "service_contracts.yml",
            "contracts.yaml",
            "dependencies.yaml",
        ]
        search_dirs = [
            self.root / "config",
            self.root,
        ]
        # Also search one level under root for common patterns
        for d in self.root.iterdir():
            if d.is_dir() and d.name in ("config", "deploy", "infra", "infrastructure"):
                search_dirs.append(d)

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for name in candidates:
                path = search_dir / name
                if path.exists():
                    data = load_yaml(str(path))
                    self.contracts = data.get("services", data)
                    break
            if self.contracts:
                break

        # Also merge dependencies from service-manifest.json if present.
        # The manifest often has more complete dependency information.
        self._merge_manifest_deps()

    def _merge_manifest_deps(self):
        """Merge dependency info from JSON service manifests into contracts.

        Searches for service-manifest.json (or similar) files that contain
        a {services: [{name, dependencies}, ...]} schema and merges their
        depends_on lists into self.contracts.
        """
        manifest_names = ["service-manifest.json", "services.json"]
        search_dirs = [self.root, self.root / "config"]
        # Also check one level deep for common layouts like project/services/
        for d in self.root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                search_dirs.append(d)
                svc_sub = d / "services"
                if svc_sub.is_dir():
                    search_dirs.append(svc_sub)

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for name in manifest_names:
                path = search_dir / name
                if not path.exists():
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                services = data.get("services", [])
                if not isinstance(services, list):
                    continue
                for entry in services:
                    svc_name = entry.get("name", "")
                    deps = entry.get("dependencies", [])
                    if svc_name and deps:
                        if svc_name not in self.contracts:
                            self.contracts[svc_name] = {}
                        existing = self.contracts[svc_name].get("depends_on", [])
                        merged = list(set(existing + deps))
                        self.contracts[svc_name]["depends_on"] = merged

    def _auto_detect_services_dir(self) -> Path:
        """Auto-detect the directory containing service source code.

        Looks for directories that contain Go/Rust/Python source files
        and have multiple subdirectories (each being a service).
        """
        # Common patterns for service directories
        candidates = [
            "services",
            "src/services",
            "cmd",
            "apps",
            "microservices",
            "internal/services",
        ]

        # Check two levels deep for the pattern: root/<prefix>/services
        for d in self.root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                for sub in candidates:
                    full = d / sub if "/" not in sub else self.root / sub
                    if full.is_dir() and self._looks_like_services_dir(full):
                        return full

        # Check direct candidates under root
        for sub in candidates:
            full = self.root / sub
            if full.is_dir() and self._looks_like_services_dir(full):
                return full

        # Fallback: use root itself if it has service-like subdirectories
        if self._looks_like_services_dir(self.root):
            return self.root

        # Last resort: just use root
        return self.root

    def _looks_like_services_dir(self, path: Path) -> bool:
        """Check if a directory looks like it contains multiple services."""
        subdirs_with_source = 0
        for d in path.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            has_source = (
                list(d.glob("*.go"))[:1] or
                list(d.glob("**/*.go"))[:1] or
                list(d.glob("*.py"))[:1] or
                list(d.glob("**/*.py"))[:1] or
                list(d.glob("*.rs"))[:1] or
                list(d.glob("src/**/*.rs"))[:1]
            )
            if has_source:
                subdirs_with_source += 1
            if subdirs_with_source >= 2:
                return True
        return False

    def resolve_port(self, port: int) -> str:
        """Resolve a port number to a service name."""
        return self.port_to_service.get(port, f"unknown:{port}")

    def resolve_service(self, name: str) -> int:
        """Resolve a service name to a port. Tries suffixes."""
        if name in self.ports:
            return self.ports[name]
        for suffix in ["-go", "-rust", "-py", ""]:
            candidate = name + suffix
            if candidate in self.ports:
                return self.ports[candidate]
        # Try stripping suffixes
        base = name.replace("-go", "").replace("-rust", "").replace("-py", "")
        for suffix in ["-go", "-rust", "-py", ""]:
            candidate = base + suffix
            if candidate in self.ports:
                return self.ports[candidate]
        return 0

    def get_declared_deps(self, service: str) -> list:
        """Get declared dependencies from contracts config."""
        contract = self.contracts.get(service, {})
        if isinstance(contract, dict):
            return contract.get("depends_on", [])
        return []

    def is_degraded_ok(self, service: str) -> bool:
        """Check if a service is allowed to be degraded."""
        contract = self.contracts.get(service, {})
        if isinstance(contract, dict):
            return contract.get("degraded_ok", False)
        return False

    def get_service_dirs(self) -> dict:
        """Find all service directories with their language."""
        result = {}
        if not self.services_dir or not self.services_dir.exists():
            return result
        for d in sorted(self.services_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # Skip archived/deprecated directories
            if d.name.startswith("_archived") or d.name.startswith("_deprecated"):
                continue
            # Detect language
            lang = "unknown"
            if list(d.glob("*.go"))[:1] or list(d.glob("**/*.go"))[:1]:
                lang = "go"
            elif list(d.glob("*.py"))[:1] or list(d.glob("**/*.py"))[:1]:
                lang = "python"
            elif list(d.glob("*.rs"))[:1] or list(d.glob("src/**/*.rs"))[:1]:
                lang = "rust"
            elif list(d.glob("*.ts"))[:1] or list(d.glob("*.js"))[:1]:
                lang = "node"
            result[d.name] = {"path": str(d), "language": lang}
        return result
