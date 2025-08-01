#!/usr/bin/env python3
"""
Generate API documentation for zignal Python bindings using pdoc.

This script generates static HTML documentation that can be hosted on GitHub Pages.

Usage:
    cd bindings/python
    uv run --extra docs python build_docs.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Generate documentation for zignal Python bindings."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    bindings_dir = script_dir.parent
    project_root = bindings_dir.parent

    # Ensure we're in the right directory
    os.chdir(project_root)

    # Check if pdoc is installed
    try:
        import pdoc
    except ImportError:
        print("Error: pdoc is not installed.")
        print("Install it with: cd bindings/python && uv pip install -e '.[docs]'")
        print("Or run with: cd bindings/python && uv run --extra docs python build_docs.py")
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
    if os.environ.get("UV_PROJECT_ROOT") or Path(bindings_dir / ".venv").exists():
        # Use uv pip for installation
        result = subprocess.run(
            ["uv", "pip", "install", "-e", str(bindings_dir)], capture_output=True
        )
    else:
        # Fall back to regular pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(bindings_dir)],
            capture_output=True,
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
    docs_dir = bindings_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Import pdoc modules
    import pdoc.doc
    import pdoc.render
    import pdoc.doc_pyi

    # Configure pdoc with critical settings for stub files
    print("Configuring pdoc...")
    pdoc.render.configure(
        show_source=False,  # CRITICAL: Must be False for stub files to work
        template_directory=None,
        search=True,
        favicon=None,
        logo=None,
        logo_link=None,
        edit_url_map=None,
        mermaid=False,
        math=False,
    )

    # Generate documentation using API
    print(f"Generating documentation in {docs_dir}...")

    # Import zignal modules to ensure they're loaded
    import zignal
    import zignal._zignal

    # Create documentation objects for both modules
    print("Creating documentation objects...")
    modules = {
        "zignal": pdoc.doc.Module(zignal),
        "zignal._zignal": pdoc.doc.Module(zignal._zignal),
    }

    # Include type information from stub files for both modules
    print("Loading type information from stub files...")

    import pdoc.doc_pyi

    # Create a temporary stub package structure that pdoc expects
    # pdoc looks for "package-stubs" directory structure per PEP-561
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create zignal-stubs directory structure
        stub_pkg_dir = Path(temp_dir) / "zignal-stubs"
        stub_pkg_dir.mkdir()

        # Create py.typed marker
        (stub_pkg_dir / "py.typed").touch()

        # Since Image.__module__ = "zignal", pdoc will look for types in zignal/__init__.pyi
        # But our __init__.pyi only has re-exports. We need to put the actual definitions there.
        # So let's use the _zignal.pyi content for the main __init__.pyi
        shutil.copy2(
            "/home/adria/Projects/zignal/bindings/python/zignal/_zignal.pyi",
            stub_pkg_dir / "__init__.pyi",  # Use _zignal.pyi content for zignal module
        )
        shutil.copy2(
            "/home/adria/Projects/zignal/bindings/python/zignal/_zignal.pyi",
            stub_pkg_dir / "_zignal.pyi",
        )

        # Add temp directory to sys.path temporarily
        import sys

        sys.path.insert(0, temp_dir)

        try:
            for module_name, module_doc in modules.items():
                print(f"  Processing {module_name}...")

                # Now pdoc should find the stub files
                # pdoc will find the stub files from the -stubs package

                # Apply typeinfo
                pdoc.doc_pyi.include_typeinfo_from_stub_files(module_doc)

        finally:
            # Remove temp directory from sys.path
            sys.path.remove(temp_dir)

        # Debug: Check if type annotations were loaded
        if module_name == "zignal" and "Image" in module_doc.members:
            image_class = module_doc.members["Image"]
            if "load" in image_class.members:
                load_method = image_class.members["load"]
                if load_method.signature:
                    print(f"    ✓ Found signature for Image.load: {load_method.signature}")
                    # Check parameters in detail
                    for param_name, param in load_method.signature.parameters.items():
                        print(
                            f"      Parameter '{param_name}': annotation={param.annotation}, kind={param.kind}"
                        )
                    # Check return annotation
                    print(f"      Return annotation: {load_method.signature.return_annotation}")
                else:
                    print(f"    ✗ No signature found for Image.load")

    # Create all_modules dict for cross-references
    all_modules = {}
    for name, doc in modules.items():
        all_modules[name] = doc

    # Generate HTML for each module
    for module_name, module_doc in modules.items():
        print(f"Generating HTML for {module_name}...")
        html = pdoc.render.html_module(module_doc, all_modules)

        # Save HTML file
        output_file = docs_dir / f"{module_name.replace('.', '/')}.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html)

    # Generate search index
    print("Generating search index...")
    search_js = pdoc.render.search_index(all_modules)
    search_file = docs_dir / "search.js"
    search_file.write_text(search_js)

    print("\nDocumentation generated successfully!")
    print("Search functionality has been enabled!")

    # Check what files were generated
    html_files = list(docs_dir.glob("**/*.html"))
    if html_files:
        print(f"\nGenerated files:")
        for f in html_files:
            print(f"  - {f.relative_to(docs_dir)}")

    print(f"\nDocumentation is available in {docs_dir}")


if __name__ == "__main__":
    main()
