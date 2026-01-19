#!/usr/bin/env python3

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(cmd, cwd=None, env=None, check=True):
    """Run a command with proper logging and error handling."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  in directory: {cwd}")

    environ = env if env is not None else os.environ.copy()
    result = subprocess.run(cmd, cwd=cwd, env=environ, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[3]


def bindings_dir() -> Path:
    return project_root() / "bindings" / "python"


def python_cmd() -> list[str]:
    """Return the command to run python via uv."""
    return ["uv", "run", "python"]


def ensure_deps(with_numpy: bool) -> None:
    """Ensure build and test dependencies are installed."""
    print("=== Ensuring Dependencies ===")
    deps = ["setuptools", "wheel", "build", "pytest"]
    if with_numpy:
        deps.append("numpy")
    if platform.system() == "Darwin":
        deps.append("delocate")

    run(["uv", "pip", "install", "--upgrade", *deps])


def update_version(skip: bool) -> None:
    """Sync version from Zig to pyproject.toml."""
    if skip:
        return

    script = bindings_dir() / "scripts" / "update_version.py"
    if script.exists():
        run([*python_cmd(), str(script)], cwd=bindings_dir())


def clean_artifacts() -> None:
    """Clean build artifacts."""
    print("=== Cleaning Artifacts ===")
    for target in ["build", "dist", "zignal.egg-info"]:
        path = bindings_dir() / target
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    for ext in ["*.so", "*.pyd", "*.dylib"]:
        for file in bindings_dir().rglob(ext):
            file.unlink()


def build_wheel(optimize: str) -> None:
    """Build the wheel using uv run python -m build."""
    print(f"=== Building Wheel (Optimize: {optimize}) ===")

    env = os.environ.copy()
    env["ZIG_OPTIMIZE"] = optimize
    env["ZIG_CPU"] = env.get("ZIG_CPU", "baseline")

    run([*python_cmd(), "-m", "build", "--wheel"], cwd=bindings_dir(), env=env)

    if platform.system() == "Darwin":
        print("=== Delocating Wheel (macOS) ===")
        dist_dir = bindings_dir() / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if wheels:
            latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
            run([*python_cmd(), "-m", "delocate.cmd.delocate_wheel", "-w", str(dist_dir), "-v", str(latest_wheel)])


def test_artifacts() -> None:
    """Install the built wheel and run tests."""
    print("=== Testing Artifacts ===")

    dist_dir = bindings_dir() / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel found to test!")

    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    run(["uv", "pip", "install", str(latest_wheel), "--force-reinstall"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copytree(bindings_dir() / "tests", tmp_path / "tests")
        run([*python_cmd(), "-m", "pytest", "tests", "-v"], cwd=tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zignal Wheel Build Helper")
    parser.add_argument("--optimize", default="ReleaseFast", choices=["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"])
    parser.add_argument("--skip-version-update", action="store_true")
    parser.add_argument("--with-numpy", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")

    args = parser.parse_args()

    ensure_deps(args.with_numpy)
    update_version(args.skip_version_update)
    clean_artifacts()
    build_wheel(args.optimize)

    if not args.skip_tests:
        test_artifacts()


if __name__ == "__main__":
    main()
