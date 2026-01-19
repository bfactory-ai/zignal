#!/usr/bin/env python3
"""
Build wheels for the current platform.

This script wraps `python -m build` to handle platform-specific details:
- Setting correct platform tags (manylinux, macosx)
- Repairing wheels on macOS (delocate)
- Managing build dependencies

Usage:
    cd bindings/python
    python scripts/build_wheels.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_platform_config():
    """Determine platform tags and Zig target for the current machine."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    plat_name = None
    zig_target = "native"

    if system == "linux":
        # Zig creates self-contained binaries, so we can claim manylinux compatibility
        if machine in ["x86_64", "amd64"]:
            plat_name = "manylinux2014_x86_64"
        elif machine in ["aarch64", "arm64"]:
            plat_name = "manylinux2014_aarch64"

    elif system == "darwin":
        if machine == "arm64":
            plat_name = "macosx_11_0_arm64"
            zig_target = "aarch64-macos-none"
        else:
            plat_name = "macosx_10_9_x86_64"
            zig_target = "x86_64-macos-none"

    elif system == "windows":
        if machine in ["x86_64", "amd64"]:
            plat_name = "win_amd64"
        elif machine in ["aarch64", "arm64"]:
            plat_name = "win_arm64"

    return plat_name, zig_target


def main():
    script_dir = Path(__file__).parent
    bindings_dir = script_dir.parent

    # Ensure we are in the bindings directory
    os.chdir(bindings_dir)

    # 1. Update version
    print("Updating version...")
    subprocess.run([sys.executable, "scripts/update_version.py"], check=True)

    # 2. Configure environment
    plat_name, zig_target = get_platform_config()
    env = os.environ.copy()

    if plat_name:
        env["PLAT_NAME"] = plat_name
        print(f"Platform tag: {plat_name}")

    if "ZIG_TARGET" not in env:
        env["ZIG_TARGET"] = zig_target
        print(f"Zig target:   {zig_target}")

    # Set defaults if not provided
    if "ZIG_OPTIMIZE" not in env:
        env["ZIG_OPTIMIZE"] = "ReleaseFast"

    if "ZIG_CPU" not in env:
        env["ZIG_CPU"] = "baseline"

    # 3. Clean previous builds
    if (bindings_dir / "dist").exists():
        shutil.rmtree(bindings_dir / "dist")

    # 4. Build wheel
    print("\nBuilding wheel...")
    cmd = [sys.executable, "-m", "build", "--wheel"]

    # We pass platform name via env var PLAT_NAME which setup.py/setuptools reads,
    # or passed as an argument if supported by the build frontend, but environment
    # is the most robust way for setuptools to pick it up.

    # However, 'build' tool calls setup.py in isolation. We need to pass arguments
    # to the build backend or rely on environment variables being propagated.
    # 'python -m build' propagates environment variables.

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        print("Build failed.")
        sys.exit(1)

    # 5. Repair wheel (macOS only)
    if platform.system() == "Darwin":
        print("\nRepairing wheel with delocate...")
        dist_dir = bindings_dir / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            print("No wheels found to repair.")
            sys.exit(1)

        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)

        try:
            subprocess.run(
                ["delocate-wheel", "-w", str(dist_dir), "-v", str(latest_wheel)],
                check=True
            )
        except FileNotFoundError:
            print("Warning: 'delocate-wheel' not found. Skipping repair.")
            print("Install it with: pip install delocate")
        except subprocess.CalledProcessError:
            print("Wheel repair failed.")
            sys.exit(1)

    print(f"\nSuccess! Wheel available in {bindings_dir}/dist")


if __name__ == "__main__":
    main()
