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

def test_detects_cors_wildcard():
    with tempfile.TemporaryDirectory() as tmp:
        svc_dir = Path(tmp) / "svc"
        svc_dir.mkdir()
        (svc_dir / "main.go").write_text('''
package main
config.AllowAllOrigins = true
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
