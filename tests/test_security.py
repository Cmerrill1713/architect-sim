from architect_sim.simulation.security_scan import check_security
from architect_sim.models import ServiceBlueprint
from pathlib import Path
import tempfile

def test_detects_hardcoded_password():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "main.go").write_text('''
package main
var dbPassword = "super_secret_password123"
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="go", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert any("Hardcoded Password" in f.endpoint for f in findings)

def test_detects_production_cors_wildcard():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "main.go").write_text('''
package main
w.Header().Set("production deploy Access-Control-Allow-Origin", "*")
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="go", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert any("CORS" in f.endpoint for f in findings)

def test_skips_test_files():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "main_test.go").write_text('''
package main
var testPassword = "test_secret_123456"
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="go", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert len(findings) == 0

def test_detects_sql_injection():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "app.py").write_text('''
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="python", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert any("SQL" in f.endpoint for f in findings)

def test_skips_safe_localhost_database_url_default():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "app.py").write_text('''
import os
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/athena")
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="python", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert not any("Database URL" in f.endpoint for f in findings)

def test_skips_hashed_path_component_join():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "app.py").write_text('''
import hashlib
import os

name = hashlib.sha256(user_input.encode()).hexdigest()
path = os.path.join("/tmp/cache", name)
''')
        bp = ServiceBlueprint(name="svc", port=8080, language="python", source_dir=str(svc_dir))
        findings = check_security({"svc": bp}, None)
        assert not any("Path Traversal" in f.endpoint for f in findings)
