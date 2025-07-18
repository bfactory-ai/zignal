"""Zignal Python bindings - zero-dependency image processing library."""

# Import the compiled extension module
import sys
import os
import glob

print(f"DEBUG: Python version: {sys.version}", file=sys.stderr)
print(f"DEBUG: __file__: {__file__}", file=sys.stderr)
print(f"DEBUG: Package directory: {os.path.dirname(__file__)}", file=sys.stderr)

# List all files in the package directory
pkg_dir = os.path.dirname(__file__)
if os.path.exists(pkg_dir):
    print(f"DEBUG: Files in package directory:", file=sys.stderr)
    for file in os.listdir(pkg_dir):
        file_path = os.path.join(pkg_dir, file)
        print(f"DEBUG:   {file} ({'file' if os.path.isfile(file_path) else 'dir'})", file=sys.stderr)

# Look for any _zignal files
zignal_files = glob.glob(os.path.join(pkg_dir, "*zignal*"))
print(f"DEBUG: Files matching *zignal*: {zignal_files}", file=sys.stderr)

# Try the import
try:
    from ._zignal import *
    print("DEBUG: Import successful", file=sys.stderr)
except ImportError as e:
    print(f"DEBUG: Import failed: {e}", file=sys.stderr)
    print(f"DEBUG: Exception type: {type(e)}", file=sys.stderr)

    # Try alternative import methods
    import importlib.util

    # Find the actual extension file
    zignal_path = None
    for file in os.listdir(pkg_dir):
        if file.startswith("_zignal.") and (file.endswith(".so") or file.endswith(".pyd") or file.endswith(".dylib")):
            zignal_path = os.path.join(pkg_dir, file)
            break

    if zignal_path and os.path.exists(zignal_path):
        print(f"DEBUG: Found extension at {zignal_path}", file=sys.stderr)
        try:
            spec = importlib.util.spec_from_file_location("_zignal", zignal_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Import everything from the module
                for name in dir(module):
                    if not name.startswith('_'):
                        globals()[name] = getattr(module, name)
                print("DEBUG: Manual import successful", file=sys.stderr)
            else:
                print("DEBUG: Could not create module spec", file=sys.stderr)
        except Exception as manual_e:
            print(f"DEBUG: Manual import failed: {manual_e}", file=sys.stderr)
            raise e
    else:
        print(f"DEBUG: Extension file not found at {zignal_path}", file=sys.stderr)
        raise e

__version__ = "0.1.3"
__all__ = ["Rgb"]
