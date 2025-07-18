#!/usr/bin/env python3
"""Build wheels for multiple platforms using local Zig cross-compilation."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# Platform configurations: (zig_target, wheel_platform_tag, extension)
PLATFORMS = [
    ("native", "linux_x86_64", ".so"),  # Native Linux
    # Cross-compilation targets (requires Python headers for target platforms)
    # ("x86_64-windows-msvc", "win_amd64", ".pyd"),
    # ("x86_64-macos-none", "macosx_10_9_x86_64", ".dylib"),
    # ("aarch64-macos-none", "macosx_11_0_arm64", ".dylib"),
]


def run_command(cmd: List[str], cwd: Path = None, env: dict = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  in directory: {cwd}")
    
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    return result


def prepare_clean_build_env(bindings_dir: Path) -> None:
    """Prepare a clean build environment by removing any precompiled extensions."""
    print("\n=== Preparing clean build environment ===")
    
    # Remove any existing built extensions from the package directory
    package_dir = bindings_dir / "zignal"
    for ext in [".so", ".dylib", ".pyd", ".dll"]:
        ext_file = package_dir / f"zignal{ext}"
        if ext_file.exists():
            print(f"Removing existing extension: {ext_file}")
            ext_file.unlink()
    
    # Clean up build directories
    for build_dir in ["build", "dist", "*.egg-info"]:
        full_path = bindings_dir / build_dir
        if full_path.exists():
            print(f"Removing build directory: {full_path}")
            if full_path.is_dir():
                import shutil
                shutil.rmtree(full_path)
            else:
                full_path.unlink()


def create_wheel(
    zig_target: str, 
    platform_tag: str, 
    extension: str,
    bindings_dir: Path
) -> Path:
    """Create a wheel for the given platform."""
    print(f"\n=== Creating wheel for {platform_tag} ===")
    
    # Set environment variables to control the build
    env = os.environ.copy()
    env["PLAT_NAME"] = platform_tag
    env["ZIG_TARGET"] = zig_target
    env["ZIG_OPTIMIZE"] = env.get("ZIG_OPTIMIZE", "ReleaseFast")  # Default to ReleaseFast
    
    # Use python setup.py directly to build and create wheel
    # The setup.py will automatically build the Zig extension
    
    # Use the same Python interpreter that's running this script
    # This ensures we use the same environment (venv or system) consistently
    python_exe = sys.executable
    
    cmd = [python_exe, "setup.py", "bdist_wheel", "--plat-name", platform_tag]
    result = run_command(cmd, cwd=bindings_dir, env=env)
    
    # Find the generated wheel
    dist_dir = bindings_dir / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel file found after build")
    
    # Get the most recent wheel
    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    
    print(f"Created wheel: {latest_wheel}")
    
    # Debug: Check wheel contents
    import zipfile
    print("DEBUG: Wheel contents:")
    with zipfile.ZipFile(latest_wheel, 'r') as zf:
        for name in zf.namelist():
            print(f"  {name}")
    
    return latest_wheel


def build_all_wheels(platforms: List[Tuple[str, str, str]]) -> List[Path]:
    """Build wheels for all specified platforms."""
    script_dir = Path(__file__).parent
    
    # Prepare clean build environment
    prepare_clean_build_env(script_dir)
    
    wheels = []
    
    for zig_target, platform_tag, expected_ext in platforms:
        try:
            # Create the wheel (setup.py will build the Zig extension automatically)
            wheel_path = create_wheel(zig_target, platform_tag, expected_ext, script_dir)
            wheels.append(wheel_path)
            
        except Exception as e:
            print(f"Failed to build wheel for {platform_tag}: {e}")
            if not args.continue_on_error:
                raise
    
    return wheels


def main():
    parser = argparse.ArgumentParser(description="Build wheels for multiple platforms")
    parser.add_argument(
        "--platforms", 
        nargs="*", 
        help="Platform tags to build (default: all supported)"
    )
    parser.add_argument(
        "--continue-on-error", 
        action="store_true",
        help="Continue building other platforms if one fails"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true", 
        help="List all supported platforms"
    )
    parser.add_argument(
        "--optimize", 
        default="ReleaseFast",
        choices=["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"],
        help="Zig optimization mode (default: ReleaseFast)"
    )
    
    global args
    args = parser.parse_args()
    
    if args.list_platforms:
        print("Supported platforms:")
        for zig_target, platform_tag, ext in PLATFORMS:
            print(f"  {platform_tag} ({zig_target})")
        return
    
    # Set optimization level from command line
    os.environ["ZIG_OPTIMIZE"] = args.optimize
    
    # Filter platforms if specific ones were requested
    platforms_to_build = PLATFORMS
    if args.platforms:
        platforms_to_build = [
            (zig, plat, ext) for zig, plat, ext in PLATFORMS 
            if plat in args.platforms
        ]
        
        if not platforms_to_build:
            print(f"No matching platforms found for: {args.platforms}")
            print("Use --list-platforms to see available platforms")
            sys.exit(1)
    
    print(f"Building wheels for {len(platforms_to_build)} platform(s) with optimization: {args.optimize}")
    
    wheels = build_all_wheels(platforms_to_build)
    
    print(f"\n=== Summary ===")
    print(f"Successfully built {len(wheels)} wheel(s):")
    for wheel in wheels:
        print(f"  {wheel}")
    
    if wheels:
        print(f"\nTo upload to PyPI:")
        print(f"  uv publish {' '.join(str(w) for w in wheels)}")


if __name__ == "__main__":
    main()