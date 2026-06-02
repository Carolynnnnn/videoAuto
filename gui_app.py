"""
DEPRECATED: gui_app.py - Legacy GUI entrypoint

This file is deprecated and maintained only for backward compatibility.

All new GUI development and features are in the canonical implementation:
    src.gui.app

To migrate to the new GUI:
    python3 -m src.gui.app

This shim will forward to the new implementation with a deprecation warning.
"""
import sys
import warnings
from pathlib import Path

APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def main():
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "DEPRECATION WARNING: gui_app.py is deprecated\n"
        "\n"
        "The legacy GUI entrypoint (gui_app.py) is deprecated and will be\n"
        "removed in a future release.\n"
        "\n"
        "Canonical GUI implementation: src.gui.app\n"
        "\n"
        "To launch the new GUI directly:\n"
        "    python3 -m src.gui.app\n"
        "\n"
        "Forwarding to new implementation...\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        from src.gui.app import main as new_gui_main
        new_gui_main()
    except ImportError as e:
        print(f"ERROR: Failed to import new GUI implementation: {e}", file=sys.stderr)
        print("\nPlease ensure src/gui/app.py exists and dependencies are installed.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: GUI launch failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
