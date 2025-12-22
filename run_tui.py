#!/usr/bin/env python3
"""
AVA TUI Entry Point
===================

Launch the AVA Terminal User Interface.

Usage:
    python run_tui.py [--backend URL] [--debug]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AVA Terminal User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default="http://localhost:8085",
        help="Backend API URL (default: http://localhost:8085)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="Start without connecting to backend (offline mode)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup if just showing help
    from tui import AVATUI

    # Create and run the app
    app = AVATUI()
    app.backend_url = args.backend
    app.debug_mode = args.debug
    app.offline_mode = args.no_connect

    # Run the application
    app.run()


if __name__ == "__main__":
    main()
