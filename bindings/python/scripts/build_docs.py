#!/usr/bin/env python3
"""
Generate API documentation for zignal Python bindings using pdoc.

Usage:
    cd bindings/python
    uv run scripts/build_docs.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    # Use absolute paths to avoid confusion during chdir
    script_path = Path(__file__).resolve()
    bindings_dir = script_path.parent.parent
    root = bindings_dir.parent.parent
    os.chdir(root)

    # 1. Build Bindings
    print("Building Python bindings...")
    try:
        subprocess.check_call(["zig", "build", "python-bindings"])
    except subprocess.CalledProcessError:
        sys.exit(1)

    # 2. Type Check with ty
    print("Validating type stubs with ty...")
    try:
        # Check the package directory which contains the .pyi stubs
        # Path is relative to the project root
        subprocess.check_call(["ty", "check", "bindings/python/zignal"])
        print("✓ Type annotations look good!")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if isinstance(e, FileNotFoundError):
            print("Warning: 'ty' not found. Skipping type validation.")
        else:
            print("\n✗ Type validation failed! Please check the stubs in bindings/python/zignal/")
            sys.exit(1)

    # 3. Generate Docs
    docs_dir = bindings_dir / "docs"
    shutil.rmtree(docs_dir, ignore_errors=True)
    docs_dir.mkdir(parents=True)

    print("Generating documentation with pdoc...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "empty.py").write_text("'''Search index placeholder.'''")

        # Set PYTHONPATH to include the dummy module for search generation
        env = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}

        try:
            subprocess.check_call(
                ["pdoc", "zignal", "empty", "-o", str(docs_dir), "--no-show-source"],
                env=env
            )
        except subprocess.CalledProcessError:
            sys.exit(1)

    print(f"\n✓ Documentation generated successfully in {docs_dir}")


if __name__ == "__main__":
    main()
