"""Setup script for zignal Python bindings."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


class ZigExtension(Extension):
    """Extension that will be built with Zig."""
    
    def __init__(self, name: str, zig_target: str = "native", zig_optimize: str = "ReleaseFast"):
        # Initialize with dummy source to satisfy setuptools
        super().__init__(name, sources=[])
        self.zig_target = zig_target
        self.zig_optimize = zig_optimize


class ZigBuildExt(build_ext):
    """Custom build_ext command that uses Zig."""
    
    def build_extension(self, ext: ZigExtension) -> None:
        if not isinstance(ext, ZigExtension):
            return super().build_extension(ext)
        
        print(f"Building Zig extension with target: {ext.zig_target}, optimize: {ext.zig_optimize}")
        
        # Find the project root (where build.zig is located)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Verify build.zig exists - we always build from source
        build_zig = project_root / "build.zig"
        if not build_zig.exists():
            raise RuntimeError(
                f"build.zig not found at {build_zig}. "
                f"The Python bindings must be built from the zignal project root directory "
                f"which contains the build.zig file and source code."
            )
        
        # Build the Zig library with optimizations
        cmd = ["zig", "build", "python-bindings", f"-Doptimize={ext.zig_optimize}"]
        if ext.zig_target != "native":
            cmd.extend([f"-Dtarget={ext.zig_target}"])
        
        print(f"Running: {' '.join(cmd)} in {project_root}")
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Zig build failed with return code {result.returncode}", file=sys.stderr)
            print(f"stdout: {result.stdout}", file=sys.stderr)
            print(f"stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Zig build failed: {result.stderr}")
        
        # Find the built library
        lib_dir = project_root / "zig-out" / "lib"
        
        # Look for the library with different possible extensions
        extensions = [".so", ".dylib", ".pyd", ".dll"]
        library_path = None
        
        for extension in extensions:
            candidate = lib_dir / f"zignal{extension}"
            if candidate.exists():
                library_path = candidate
                break
        
        if not library_path:
            raise RuntimeError(f"Built library not found in {lib_dir}")
        
        # Determine destination path
        dest_dir = Path(self.get_ext_fullpath(ext.name)).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy to the correct location with correct name
        dest_path = Path(self.get_ext_fullpath(ext.name))
        print(f"Copying {library_path} -> {dest_path}")
        shutil.copy2(library_path, dest_path)


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform tag."""
    
    def has_ext_modules(self):
        return True


def get_zig_target():
    """Get Zig target from environment variable, defaulting to native."""
    return os.environ.get("ZIG_TARGET", "native")


def get_zig_optimize():
    """Get Zig optimization mode from environment variable, defaulting to ReleaseFast."""
    return os.environ.get("ZIG_OPTIMIZE", "ReleaseFast")


if __name__ == "__main__":
    setup(
        packages=find_packages(),
        ext_modules=[ZigExtension("zignal.zignal", get_zig_target(), get_zig_optimize())],
        cmdclass={"build_ext": ZigBuildExt},
        distclass=BinaryDistribution,
        zip_safe=False,
    )