"""Configuration loader with auto-detection for any microservice project.

Supports:
  Port configs:    ports.yaml, docker-compose.yml, service-manifest.json,
                   kubernetes manifests, .env files, package.json proxy,
                   nginx.conf, Caddyfile, Procfile, turbo.json
  Contracts:       service_contracts.yaml, dependencies.yaml
  Service dirs:    auto-detected from source file patterns
"""
import json
import os
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: str) -> dict:
    """Load YAML file."""
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

        # Supplement with port references found in source code
        self._scan_source_for_ports()

        # Load service contracts if available
        self._auto_detect_contracts()

        # Auto-detect services directory if not specified
        if self.services_dir is None:
            self.services_dir = self._auto_detect_services_dir()

    # ================================================================
    # PORT CONFIG LOADING — supports many formats
    # ================================================================

    def _load_port_config(self, path: Path):
        """Load port configuration from a specific file."""
        if not path.exists():
            return

        name = path.name.lower()
        suffix = path.suffix.lower()

        if name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            self._load_docker_compose(path)
        elif suffix in (".yaml", ".yml"):
            self._load_yaml_ports(path)
        elif suffix == ".json":
            self._load_json_ports(path)
        elif name == ".env" or name.startswith(".env."):
            self._load_env_ports(path)
        elif name in ("procfile", "procfile.dev"):
            self._load_procfile(path)
        elif name in ("nginx.conf",):
            self._load_nginx(path)
        elif name in ("caddyfile",):
            self._load_caddyfile(path)

    def _load_yaml_ports(self, path: Path):
        """Load ports from a YAML file. Supports multiple schemas."""
        data = load_yaml(str(path))

        # Schema 1: ports.yaml — {ports: {name: {port: N, ...}}}
        if "ports" in data and isinstance(data["ports"], dict):
            for name, info in data["ports"].items():
                if isinstance(info, dict):
                    port = info.get("port")
                    if port:
                        self._register_port(name, int(port))
                elif isinstance(info, int):
                    self._register_port(name, info)
            return

        # Schema 2: Kubernetes Service — {kind: Service, spec: {ports: [{port: N}]}}
        if data.get("kind") == "Service":
            svc_name = data.get("metadata", {}).get("name", "")
            for port_spec in data.get("spec", {}).get("ports", []):
                port = port_spec.get("port") or port_spec.get("targetPort")
                if svc_name and port:
                    self._register_port(svc_name, int(port))
            return

        # Schema 3: Kubernetes Deployment with container ports
        if data.get("kind") in ("Deployment", "StatefulSet", "DaemonSet"):
            svc_name = data.get("metadata", {}).get("name", "")
            containers = data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            for container in containers:
                for port_spec in container.get("ports", []):
                    port = port_spec.get("containerPort")
                    if port:
                        name = svc_name or container.get("name", "")
                        if name:
                            self._register_port(name, int(port))
            return

        # Schema 4: Helm values — {services: {name: {port: N}}} or similar
        if "services" in data and isinstance(data["services"], dict):
            for name, info in data["services"].items():
                if isinstance(info, dict) and "port" in info:
                    self._register_port(name, int(info["port"]))
            return

        # Schema 5: Flat {service_name: port_number} or {service_name: {port: N}}
        for name, val in data.items():
            if isinstance(val, int) and 1024 < val < 65535:
                self._register_port(name, val)
            elif isinstance(val, dict) and "port" in val:
                self._register_port(name, int(val["port"]))

    def _load_docker_compose(self, path: Path):
        """Extract port mappings from docker-compose.yml."""
        data = load_yaml(str(path))
        services = data.get("services", {})
        for name, svc_config in services.items():
            # Port mappings
            ports = svc_config.get("ports", [])
            for port_mapping in ports:
                port_str = str(port_mapping)
                match = re.match(r'(\d+):(\d+)', port_str)
                if match:
                    self._register_port(name, int(match.group(1)))
                elif port_str.isdigit():
                    self._register_port(name, int(port_str))

            # Environment variables with PORT
            env = svc_config.get("environment", {})
            if isinstance(env, list):
                for item in env:
                    if "=" in str(item):
                        k, v = str(item).split("=", 1)
                        if "PORT" in k and v.isdigit():
                            self._register_port(name, int(v))
            elif isinstance(env, dict):
                for k, v in env.items():
                    if "PORT" in k and str(v).isdigit():
                        self._register_port(name, int(v))

    def _load_json_ports(self, path: Path):
        """Load ports from a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        name = path.name.lower()

        # package.json — proxy field or scripts with port
        if name == "package.json":
            proxy = data.get("proxy")
            if proxy and isinstance(proxy, str):
                match = re.search(r':(\d+)', proxy)
                if match:
                    self._register_port("proxy-target", int(match.group(1)))
            # Scripts with --port
            scripts = data.get("scripts", {})
            for script_name, cmd in scripts.items():
                match = re.search(r'--port[= ](\d+)', str(cmd))
                if match:
                    self._register_port(script_name, int(match.group(1)))
            return

        # turbo.json — pipeline with port info
        if name == "turbo.json":
            pipeline = data.get("pipeline", data.get("tasks", {}))
            for task_name, task_config in pipeline.items():
                if isinstance(task_config, dict):
                    env = task_config.get("env", [])
                    for e in env:
                        if "PORT" in str(e):
                            match = re.search(r'(\d{4,5})', str(e))
                            if match:
                                self._register_port(task_name.replace("#", "-"), int(match.group(1)))
            return

        # service-manifest format: {services: [{name, port}, ...]}
        if "services" in data and isinstance(data["services"], list):
            for svc in data["services"]:
                name = svc.get("name", "")
                port = svc.get("port", 0)
                if name and port:
                    self._register_port(name, int(port))
            return

        # Flat dict format
        for name, val in data.items():
            if isinstance(val, int) and 1024 < val < 65535:
                self._register_port(name, val)

    def _load_env_ports(self, path: Path):
        """Extract port assignments from .env files."""
        try:
            content = path.read_text(errors="replace")
        except (OSError, IOError):
            return

        # Match: SERVICE_PORT=8080 or PORT=3000
        for match in re.finditer(r'^(\w*PORT\w*)\s*=\s*(\d+)', content, re.MULTILINE):
            key, port_str = match.groups()
            port = int(port_str)
            if 1024 < port < 65535:
                # Derive service name: USERS_SERVICE_PORT → users-service
                svc_name = key.replace("_PORT", "").replace("_", "-").lower()
                if svc_name == "port":
                    svc_name = path.parent.name  # Use directory name
                self._register_port(svc_name, port)

    def _load_procfile(self, path: Path):
        """Extract ports from Procfile (Heroku/Foreman format)."""
        try:
            content = path.read_text(errors="replace")
        except (OSError, IOError):
            return

        # Match: web: node server.js --port 3000
        for match in re.finditer(r'^(\w+):\s*(.+)', content, re.MULTILINE):
            name, cmd = match.groups()
            port_match = re.search(r'(?:--port|PORT=|-p\s+)(\d+)', cmd)
            if port_match:
                self._register_port(name, int(port_match.group(1)))

    def _load_nginx(self, path: Path):
        """Extract upstream/proxy ports from nginx.conf."""
        try:
            content = path.read_text(errors="replace")
        except (OSError, IOError):
            return

        # upstream blocks: upstream backend { server 127.0.0.1:8080; }
        for match in re.finditer(r'upstream\s+(\w+)\s*\{[^}]*server\s+[\w.]+:(\d+)', content, re.DOTALL):
            name, port = match.groups()
            self._register_port(name, int(port))

        # proxy_pass: proxy_pass http://localhost:8080;
        for match in re.finditer(r'proxy_pass\s+https?://[\w.]+:(\d+)(/\S*)?;', content):
            port = int(match.group(1))
            if port not in self.port_to_service:
                self._register_port(f"upstream-{port}", port)

        # listen directive: listen 8080;
        for match in re.finditer(r'listen\s+(\d+)', content):
            port = int(match.group(1))
            if port not in self.port_to_service and 1024 < port < 65535:
                self._register_port("nginx", port)

    def _load_caddyfile(self, path: Path):
        """Extract ports from Caddyfile."""
        try:
            content = path.read_text(errors="replace")
        except (OSError, IOError):
            return

        # reverse_proxy localhost:8080
        for match in re.finditer(r'reverse_proxy\s+[\w.]*:?(\d+)', content):
            port = int(match.group(1))
            if port not in self.port_to_service:
                self._register_port(f"caddy-upstream-{port}", port)

        # :8080 { ... }
        for match in re.finditer(r'^:(\d+)\s*\{', content, re.MULTILINE):
            port = int(match.group(1))
            self._register_port("caddy", port)

    def _scan_source_for_ports(self):
        """Scan source code for port references not in config files.

        Finds hardcoded ports in Go main.go, Python app.py, etc.
        Only looks at top-level files to stay fast.
        """
        if not self.services_dir or not self.services_dir.exists():
            return

        port_re = re.compile(r'(?:listen|Listen|PORT|port|addr).*?[:\s=](\d{4,5})')

        for d in self.services_dir.iterdir():
            if not d.is_dir() or d.name.startswith(".") or d.name.startswith("_"):
                continue

            # Already known
            if d.name in self.ports:
                continue

            # Check main entry point files for port declarations
            for entry_file in ["main.go", "server.go", "app.py", "server.py",
                               "main.rs", "main.ts", "server.ts", "index.ts",
                               "app.ts", "index.js", "server.js", "app.js"]:
                filepath = d / entry_file
                if not filepath.exists():
                    continue
                try:
                    content = filepath.read_text(errors="replace")
                    for match in port_re.finditer(content):
                        port = int(match.group(1))
                        if 1024 < port < 65535 and port not in self.port_to_service:
                            self._register_port(d.name, port)
                            break
                except (OSError, IOError):
                    continue

    # ================================================================
    # CONTRACT / DEPENDENCY DETECTION
    # ================================================================

    def _auto_detect_contracts(self):
        """Auto-detect service contracts/dependency declarations."""
        candidates = [
            "service_contracts.yaml", "service_contracts.yml",
            "contracts.yaml", "contracts.yml",
            "dependencies.yaml", "dependencies.yml",
            "services.yaml", "services.yml",
        ]
        search_dirs = [self.root / "config", self.root]
        for d in self.root.iterdir():
            if d.is_dir() and d.name in ("config", "deploy", "infra", "infrastructure", "k8s", "helm"):
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

        self._merge_manifest_deps()

    def _merge_manifest_deps(self):
        """Merge dependency info from JSON service manifests."""
        manifest_names = ["service-manifest.json", "services.json", "manifest.json"]
        search_dirs = [self.root, self.root / "config"]
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
                    deps = entry.get("dependencies", entry.get("depends_on", []))
                    if svc_name and deps:
                        if svc_name not in self.contracts:
                            self.contracts[svc_name] = {}
                        existing = self.contracts[svc_name].get("depends_on", [])
                        merged = list(set(existing + deps))
                        self.contracts[svc_name]["depends_on"] = merged

    # ================================================================
    # AUTO-DETECT PORT CONFIG
    # ================================================================

    def _auto_detect_port_config(self):
        """Auto-detect port configuration files in the project."""
        candidates = [
            # YAML port configs
            self.root / "config" / "ports.yaml",
            self.root / "config" / "ports.yml",
            self.root / "ports.yaml",
            self.root / "ports.yml",
            # Docker Compose
            self.root / "docker-compose.yml",
            self.root / "docker-compose.yaml",
            self.root / "compose.yml",
            self.root / "compose.yaml",
            # JSON configs
            self.root / "service-manifest.json",
            self.root / "config" / "service-manifest.json",
            self.root / "manifest.json",
            # Kubernetes
            self.root / "k8s",
            self.root / "kubernetes",
            self.root / "deploy",
            self.root / "helm",
            # Env files
            self.root / ".env",
            self.root / ".env.local",
            self.root / ".env.development",
            # Procfile
            self.root / "Procfile",
            self.root / "Procfile.dev",
            # Reverse proxies
            self.root / "nginx.conf",
            self.root / "config" / "nginx.conf",
            self.root / "Caddyfile",
            # Monorepo configs
            self.root / "turbo.json",
            self.root / "package.json",
        ]

        for candidate in candidates:
            if not candidate.exists():
                continue

            if candidate.is_dir():
                # Scan directory for k8s manifests
                self._scan_k8s_dir(candidate)
            else:
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

    def _scan_k8s_dir(self, path: Path):
        """Scan a directory for Kubernetes manifests with port definitions."""
        for f in sorted(path.rglob("*.yaml")):
            try:
                self._load_yaml_ports(f)
            except Exception:
                continue
        for f in sorted(path.rglob("*.yml")):
            try:
                self._load_yaml_ports(f)
            except Exception:
                continue

    # ================================================================
    # SERVICE DIRECTORY DETECTION
    # ================================================================

    def _auto_detect_services_dir(self) -> Path:
        """Auto-detect the directory containing service source code."""
        candidates = [
            "services", "src/services", "apps", "microservices",
            "internal/services", "cmd", "packages", "modules",
        ]

        # Check two levels deep, preferring dirs with more service subdirs
        all_matches = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.name):
            if d.is_dir() and not d.name.startswith("."):
                for sub in candidates:
                    full = d / sub if "/" not in sub else self.root / sub
                    if full.is_dir() and self._looks_like_services_dir(full):
                        # Count how many service subdirs — prefer the one with more
                        svc_count = sum(1 for sd in full.iterdir()
                                       if sd.is_dir() and not sd.name.startswith(".")
                                       and self._detect_language(sd) != "unknown")
                        all_matches.append((svc_count, full))

        for sub in candidates:
            full = self.root / sub
            if full.is_dir() and self._looks_like_services_dir(full):
                svc_count = sum(1 for sd in full.iterdir()
                               if sd.is_dir() and not sd.name.startswith(".")
                               and self._detect_language(sd) != "unknown")
                all_matches.append((svc_count, full))

        if all_matches:
            # Return the directory with the most service subdirs
            all_matches.sort(key=lambda x: x[0], reverse=True)
            return all_matches[0][1]

        if self._looks_like_services_dir(self.root):
            return self.root

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
                list(d.glob("src/**/*.rs"))[:1] or
                list(d.glob("*.ts"))[:1] or
                list(d.glob("src/**/*.ts"))[:1] or
                list(d.glob("*.tsx"))[:1]
            )
            if has_source:
                subdirs_with_source += 1
            if subdirs_with_source >= 2:
                return True
        return False

    # ================================================================
    # RESOLUTION HELPERS
    # ================================================================

    def _register_port(self, name: str, port: int):
        """Register a port mapping (deduplicates)."""
        if port not in self.port_to_service:
            self.port_to_service[port] = name
        if name not in self.ports:
            self.ports[name] = port

    def resolve_port(self, port: int) -> str:
        """Resolve a port number to a service name."""
        return self.port_to_service.get(port, f"unknown:{port}")

    def resolve_service(self, name: str) -> int:
        """Resolve a service name to a port. Tries suffixes."""
        if name in self.ports:
            return self.ports[name]
        for suffix in ["-go", "-rust", "-py", "-service", "-svc", "-api", ""]:
            candidate = name + suffix
            if candidate in self.ports:
                return self.ports[candidate]
        # Try stripping suffixes
        base = name
        for suffix in ["-go", "-rust", "-py", "-service", "-svc", "-api"]:
            base = base.replace(suffix, "")
        for suffix in ["-go", "-rust", "-py", "-service", "-svc", "-api", ""]:
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
        """Find all service directories with their language.

        Scans the primary services_dir plus sibling directories that
        contain source files (catches UI/frontend dirs outside services/).
        """
        result = {}
        if not self.services_dir or not self.services_dir.exists():
            return result

        for d in sorted(self.services_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if d.name.startswith("_archived") or d.name.startswith("_deprecated"):
                continue
            lang = self._detect_language(d)
            if lang != "unknown":
                result[d.name] = {"path": str(d), "language": lang}

        # Scan sibling directories for UI/frontend/app dirs
        parent = self.services_dir.parent
        if parent.exists():
            skip = {"node_modules", "vendor", "dist", "build", "bin", "logs",
                    "data", "config", "docs", "scripts", "tools", "tests", "test"}
            for d in sorted(parent.iterdir()):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                if d == self.services_dir or d.name in skip:
                    continue
                lang = self._detect_language(d)
                if lang == "unknown":
                    continue
                if d.name in result and result[d.name]["language"] != lang:
                    result[f"{d.name}-{lang}"] = {"path": str(d), "language": lang}
                elif d.name not in result:
                    result[d.name] = {"path": str(d), "language": lang}

        return result

    def _detect_language(self, d: Path) -> str:
        """Detect the primary language by counting source files per language."""
        counts = {"go": 0, "rust": 0, "node": 0, "python": 0}

        for f in d.rglob("*.go"):
            if "vendor" not in str(f):
                counts["go"] += 1
                if counts["go"] > 20:
                    break
        if (d / "Cargo.toml").exists():
            counts["rust"] = 100
        else:
            for f in d.rglob("*.rs"):
                counts["rust"] += 1
                if counts["rust"] > 20:
                    break
        for ext in ("*.ts", "*.tsx", "*.jsx"):
            for f in d.rglob(ext):
                if not any(s in str(f) for s in ["node_modules", "/dist/", "/build/", "/.next/"]):
                    counts["node"] += 1
                    if counts["node"] > 20:
                        break
        for f in d.glob("src/**/*.js"):
            counts["node"] += 1
        if (d / "tsconfig.json").exists():
            counts["node"] += 10
        elif (d / "package.json").exists() and (d / "src").is_dir():
            counts["node"] += 5
        for f in d.rglob("*.py"):
            if "venv" not in str(f) and "site-packages" not in str(f):
                counts["python"] += 1
                if counts["python"] > 20:
                    break

        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else "unknown"
