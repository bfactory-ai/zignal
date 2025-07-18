"""Zignal Python bindings - zero-dependency image processing library."""

# Import the compiled extension module
import sys
import os
import importlib.util

# Try the import
try:
    from ._zignal import *
except ImportError as e:
    # Try alternative import methods for development/debugging
    pkg_dir = os.path.dirname(__file__)

    # Find the actual extension file
    zignal_path = None
    for file in os.listdir(pkg_dir):
        if file.startswith("_zignal.") and (file.endswith(".so") or file.endswith(".pyd") or file.endswith(".dylib")):
            zignal_path = os.path.join(pkg_dir, file)
            break

    if zignal_path and os.path.exists(zignal_path):
        try:
            spec = importlib.util.spec_from_file_location("_zignal", zignal_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Import everything from the module
                for name in dir(module):
                    if not name.startswith('_'):
                        globals()[name] = getattr(module, name)
            else:
                raise ImportError("Could not create module spec")
        except Exception as manual_e:
            raise ImportError(f"Failed to load _zignal extension: {manual_e}") from e
    else:
        raise ImportError(f"_zignal extension not found in {pkg_dir}") from e

__version__ = "0.1.0"
__all__ = ["Rgb", "ImageRgb"]
