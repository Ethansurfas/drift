# tests/test_cli.py
import subprocess
import sys
import os
import tempfile

DRIFT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_drift(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "drift.cli"] + args,
        capture_output=True, text=True, cwd=DRIFT_DIR,
    )


def test_drift_no_args():
    result = run_drift([])
    assert result.returncode != 0
    assert "usage" in result.stderr.lower() or "Usage" in result.stderr


def test_drift_unknown_command():
    result = run_drift(["foobar"])
    assert result.returncode != 0


def test_drift_check_valid():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False) as f:
        f.write('print "Hello, Drift!"')
        f.flush()
        result = run_drift(["check", f.name])
    os.unlink(f.name)
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_drift_check_missing_file():
    result = run_drift(["check", "/nonexistent/file.drift"])
    assert result.returncode != 0


def test_drift_build():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('x = 42\nprint "hello"')
        f.flush()
        result = run_drift(["build", f.name])
        py_path = f.name.replace(".drift", ".py")
    os.unlink(f.name)
    assert result.returncode == 0
    assert os.path.exists(py_path)
    # Verify the output is valid Python
    with open(py_path) as pf:
        content = pf.read()
    assert "import drift_runtime" in content
    os.unlink(py_path)


def test_drift_build_output_name():
    with tempfile.NamedTemporaryFile(suffix=".drift", mode="w", delete=False, dir=DRIFT_DIR) as f:
        f.write('x = 1')
        f.flush()
        result = run_drift(["build", f.name])
        py_path = f.name.replace(".drift", ".py")
    os.unlink(f.name)
    assert result.returncode == 0
    assert "Built:" in result.stdout
    if os.path.exists(py_path):
        os.unlink(py_path)


def test_drift_check_no_file_arg():
    result = run_drift(["check"])
    assert result.returncode != 0
