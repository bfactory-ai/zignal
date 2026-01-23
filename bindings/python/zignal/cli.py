import sys
import os
import subprocess
import signal

def main():
    """Wrapper to run the native zignal CLI binary."""
    # Find the binary relative to this file
    base_dir = os.path.dirname(__file__)
    binary_name = "zignal.exe" if sys.platform == "win32" else "zignal"
    binary_path = os.path.join(base_dir, binary_name)
    
    if not os.path.exists(binary_path):
        # Fallback: check if it's in the PATH (unlikely for wheel, but good for dev)
        # or maybe the user hasn't built it yet.
        print(f"Error: Could not find zignal binary at {binary_path}", file=sys.stderr)
        print("Please ensure the package is installed correctly.", file=sys.stderr)
        sys.exit(1)
        
    # Pass all arguments through to the binary
    args = [binary_path] + sys.argv[1:]
    
    # Handle signals gracefully
    try:
        # On Unix, we can use os.execv to replace the process, saving memory and signals
        if sys.platform != "win32":
            os.execv(binary_path, args)
        else:
            # On Windows, use subprocess
            sys.exit(subprocess.call(args))
    except KeyboardInterrupt:
        # Allow the subprocess to handle it or exit cleanly
        sys.exit(130)
    except OSError as e:
        print(f"Error executing {binary_name}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
