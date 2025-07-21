#!/usr/bin/env python3
"""
Generate API documentation for zignal Python bindings using pdoc.

This script generates static HTML documentation that can be hosted on GitHub Pages.

Usage:
    cd bindings/python
    uv run --extra docs python generate_docs.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Generate documentation for zignal Python bindings."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Ensure we're in the right directory
    os.chdir(project_root)
    
    # Check if pdoc is installed
    try:
        import pdoc
    except ImportError:
        print("Error: pdoc is not installed.")
        print("Install it with: cd bindings/python && uv pip install -e '.[docs]'")
        print("Or run with: cd bindings/python && uv run --extra docs python generate_docs.py")
        sys.exit(1)
    
    # Build the Python bindings first
    print("Building Python bindings...")
    result = subprocess.run(["zig", "build", "python-bindings"], capture_output=True)
    if result.returncode != 0:
        print("Error building Python bindings:")
        print(result.stderr.decode())
        sys.exit(1)
    
    # Install the package in development mode
    print("Installing zignal package in development mode...")
    # Check if we're in a uv environment
    if os.environ.get("UV_PROJECT_ROOT") or Path(script_dir / ".venv").exists():
        # Use uv pip for installation
        result = subprocess.run(
            ["uv", "pip", "install", "-e", str(script_dir)],
            capture_output=True
        )
    else:
        # Fall back to regular pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(script_dir)],
            capture_output=True
        )
    
    if result.returncode != 0:
        print("Error installing package:")
        print(result.stderr.decode())
        sys.exit(1)
    
    # Import to verify it works
    try:
        import zignal
        print(f"Successfully imported zignal version {zignal.__version__}")
    except ImportError as e:
        print(f"Error importing zignal: {e}")
        print("Make sure the Python bindings are built correctly.")
        sys.exit(1)
    
    # Create docs directory if it doesn't exist
    docs_dir = project_root / "docs" / "python"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate documentation
    print(f"Generating documentation in {docs_dir}...")
    cmd = [
        sys.executable, "-m", "pdoc",
        "--output-directory", str(docs_dir),
        "--no-show",  # Don't open browser
        "zignal"  # Use the installed package
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("Error generating documentation:")
        print(result.stderr.decode())
        sys.exit(1)
    
    print("Documentation generated successfully!")
    
    # Find the generated HTML file (pdoc3 generates based on module name)
    # Our module is named _zignal, but pdoc3 might rename it
    possible_names = ["_zignal.html", "zignal.html"]
    doc_file = None
    for name in possible_names:
        if (docs_dir / name).exists():
            doc_file = name
            break
    
    if not doc_file:
        # Look for any HTML file that's not index.html
        html_files = [f for f in docs_dir.glob("*.html") if f.name != "index.html"]
        if html_files:
            doc_file = html_files[0].name
        else:
            doc_file = "zignal.html"  # fallback
    
    print(f"Documentation file: {doc_file}")
    
    # Optional: Generate a simple index.html redirect
    index_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Zignal Python Documentation</title>
    <meta http-equiv="refresh" content="0; url={doc_file}">
</head>
<body>
    <p>Redirecting to <a href="{doc_file}">zignal documentation</a>...</p>
</body>
</html>
"""
    
    with open(docs_dir / "index.html", "w") as f:
        f.write(index_content)
    
    print(f"Created index.html redirect in {docs_dir}")


if __name__ == "__main__":
    main()