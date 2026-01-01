#!/usr/bin/env python3
"""
AVA Server - Standalone Backend
================================

A robust, standalone HTTP server for the AVA Neural Interface.
This is the primary entry point for production deployments.

Usage:
  python ava_server.py                    # Default: localhost:8085
  python ava_server.py --port 8080        # Custom port
  python ava_server.py --host 0.0.0.0     # Allow remote connections
  python ava_server.py --check            # Verify dependencies and exit
  python ava_server.py --verbose          # Extra debug logging
  python ava_server.py --log-file app.log # Log to file

Environment Variables:
  OLLAMA_HOST=http://localhost:11434      # Ollama endpoint
  AVA_LOG_LEVEL=INFO                      # Logging verbosity (DEBUG, INFO, WARNING, ERROR)
  AVA_DATA_DIR=./data                     # Data directory for persistence
  AVA_CONFIG=./config/cortex_medulla.yaml # Configuration file path

Requirements:
  - Python 3.10+
  - Ollama running with gemma3:4b model
  - Dependencies from requirements.txt

For more information: https://github.com/NAME0x0/AVA
"""

import argparse
import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Version info
__version__ = "3.3.3"
__app_name__ = "AVA Server"

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Attempt imports with graceful failures
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: str = "INFO", log_file: str | None = None, verbose: bool = False):
    """Configure comprehensive logging with optional file output."""
    if verbose:
        level = "DEBUG"

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Format with timestamp, level, and logger name
    log_format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    return logging.getLogger("ava-server")


# =============================================================================
# Preflight Checks
# =============================================================================

@dataclass
class PreflightResult:
    """Result of a single preflight check."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Complete preflight check report."""
    all_passed: bool
    critical_failed: bool
    checks: list[PreflightResult]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


def check_python_version() -> PreflightResult:
    """Check Python version is 3.10+."""
    version = sys.version_info
    passed = version >= (3, 10)
    return PreflightResult(
        name="Python Version",
        passed=passed,
        message=f"Python {version.major}.{version.minor}.{version.micro}" +
                ("" if passed else " (requires 3.10+)"),
        details={"version": f"{version.major}.{version.minor}.{version.micro}"}
    )


def check_dependencies() -> PreflightResult:
    """Check required Python packages are available."""
    missing = []
    available = []

    packages = {
        "aiohttp": AIOHTTP_AVAILABLE,
        "httpx": HTTPX_AVAILABLE,
        "yaml (pyyaml)": YAML_AVAILABLE,
    }

    for pkg, is_available in packages.items():
        if is_available:
            available.append(pkg)
        else:
            missing.append(pkg)

    passed = len(missing) == 0
    return PreflightResult(
        name="Dependencies",
        passed=passed,
        message=f"{len(available)} available" + (f", {len(missing)} missing: {', '.join(missing)}" if missing else ""),
        details={"available": available, "missing": missing}
    )


async def check_ollama(host: str = "http://localhost:11434") -> PreflightResult:
    """Check if Ollama is running and accessible."""
    if not HTTPX_AVAILABLE:
        return PreflightResult(
            name="Ollama Connection",
            passed=False,
            message="Cannot check - httpx not installed",
            details={"error": "httpx required"}
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try the root endpoint first
            resp = await client.get(f"{host}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "unknown") for m in data.get("models", [])]
                return PreflightResult(
                    name="Ollama Connection",
                    passed=True,
                    message=f"Connected, {len(models)} model(s) available",
                    details={"host": host, "models": models[:5]}  # First 5 models
                )
            else:
                return PreflightResult(
                    name="Ollama Connection",
                    passed=False,
                    message=f"Ollama returned status {resp.status_code}",
                    details={"host": host, "status": resp.status_code}
                )
    except httpx.ConnectError:
        return PreflightResult(
            name="Ollama Connection",
            passed=False,
            message=f"Cannot connect to {host}. Is Ollama running?",
            details={"host": host, "error": "Connection refused"}
        )
    except Exception as e:
        return PreflightResult(
            name="Ollama Connection",
            passed=False,
            message=f"Error: {type(e).__name__}",
            details={"host": host, "error": str(e)}
        )


async def check_ollama_models(host: str = "http://localhost:11434", required: list[str] = None) -> PreflightResult:
    """Check if required Ollama models are available."""
    if required is None:
        required = ["gemma3:4b", "gemma2:2b"]  # Fallback models

    if not HTTPX_AVAILABLE:
        return PreflightResult(
            name="Ollama Models",
            passed=False,
            message="Cannot check - httpx not installed",
            details={"error": "httpx required"}
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{host}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                available_models = [m.get("name", "") for m in data.get("models", [])]

                # Check for any of the required models (partial match)
                found_model = None
                for req in required:
                    for avail in available_models:
                        if req in avail or avail in req:
                            found_model = avail
                            break
                    if found_model:
                        break

                if found_model:
                    return PreflightResult(
                        name="Ollama Models",
                        passed=True,
                        message=f"Found: {found_model}",
                        details={"found": found_model, "available": available_models[:5]}
                    )
                else:
                    return PreflightResult(
                        name="Ollama Models",
                        passed=False,
                        message=f"No required model found. Run: ollama pull {required[0]}",
                        details={"required": required, "available": available_models}
                    )
            else:
                return PreflightResult(
                    name="Ollama Models",
                    passed=False,
                    message="Could not list models",
                    details={"status": resp.status_code}
                )
    except Exception as e:
        return PreflightResult(
            name="Ollama Models",
            passed=False,
            message=f"Error checking models: {type(e).__name__}",
            details={"error": str(e)}
        )


def check_config_file(config_path: str | None = None) -> PreflightResult:
    """Check if configuration file exists and is readable."""
    if config_path is None:
        # Search standard locations
        candidates = [
            PROJECT_ROOT / "config" / "cortex_medulla.yaml",
            PROJECT_ROOT / "config" / "ava.yaml",
            PROJECT_ROOT / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is None:
        return PreflightResult(
            name="Configuration",
            passed=True,  # Config is optional
            message="Using defaults (no config file found)",
            details={"searched": [str(c) for c in candidates]}
        )

    path = Path(config_path)
    if not path.exists():
        return PreflightResult(
            name="Configuration",
            passed=True,  # Config is optional
            message=f"Config not found: {config_path}. Using defaults.",
            details={"path": config_path, "exists": False}
        )

    if not YAML_AVAILABLE:
        return PreflightResult(
            name="Configuration",
            passed=True,
            message=f"Found {path.name} but YAML not installed",
            details={"path": config_path}
        )

    try:
        with open(path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return PreflightResult(
            name="Configuration",
            passed=True,
            message=f"Loaded {path.name}",
            details={"path": config_path, "keys": list(config.keys()) if config else []}
        )
    except Exception as e:
        return PreflightResult(
            name="Configuration",
            passed=False,
            message=f"Error reading config: {e}",
            details={"path": config_path, "error": str(e)}
        )


def check_data_dir(data_dir: str | None = None) -> PreflightResult:
    """Check if data directory is writable."""
    if data_dir is None:
        data_dir = os.environ.get("AVA_DATA_DIR", str(PROJECT_ROOT / "data"))

    path = Path(data_dir)

    try:
        path.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()

        return PreflightResult(
            name="Data Directory",
            passed=True,
            message=f"Writable: {path}",
            details={"path": str(path), "writable": True}
        )
    except Exception as e:
        return PreflightResult(
            name="Data Directory",
            passed=False,
            message=f"Cannot write to {path}: {e}",
            details={"path": str(path), "error": str(e)}
        )


def check_port_available(host: str, port: int) -> PreflightResult:
    """Check if the specified port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port))
        sock.close()

        if result == 0:
            # Port is in use
            return PreflightResult(
                name="Port Availability",
                passed=False,
                message=f"Port {port} is already in use",
                details={"host": host, "port": port, "in_use": True}
            )
        else:
            return PreflightResult(
                name="Port Availability",
                passed=True,
                message=f"Port {port} is available",
                details={"host": host, "port": port, "in_use": False}
            )
    except Exception as e:
        return PreflightResult(
            name="Port Availability",
            passed=True,  # Assume available if we can't check
            message=f"Could not verify port {port}: {e}",
            details={"host": host, "port": port, "error": str(e)}
        )


async def run_preflight_checks(
    host: str = "127.0.0.1",
    port: int = 8085,
    ollama_host: str = "http://localhost:11434",
    config_path: str | None = None,
    data_dir: str | None = None
) -> PreflightReport:
    """Run all preflight checks and return a report."""
    checks = [
        check_python_version(),
        check_dependencies(),
        await check_ollama(ollama_host),
        await check_ollama_models(ollama_host),
        check_config_file(config_path),
        check_data_dir(data_dir),
        check_port_available(host, port),
    ]

    all_passed = all(c.passed for c in checks)
    # Critical failures: dependencies or port
    critical_failed = not checks[1].passed or not checks[6].passed

    return PreflightReport(
        all_passed=all_passed,
        critical_failed=critical_failed,
        checks=checks
    )


def print_preflight_report(report: PreflightReport, logger: logging.Logger):
    """Print preflight report in a formatted way."""
    logger.info("=" * 60)
    logger.info("PREFLIGHT CHECKS")
    logger.info("=" * 60)

    for check in report.checks:
        status = "[PASS]" if check.passed else "[FAIL]"
        level = logging.INFO if check.passed else logging.WARNING
        logger.log(level, f"{status} {check.name}: {check.message}")

    logger.info("-" * 60)

    if report.all_passed:
        logger.info("All checks passed. Server ready to start.")
    elif report.critical_failed:
        logger.error("Critical checks failed. Cannot start server.")
    else:
        logger.warning("Some checks failed (non-critical). Server will start.")

    logger.info("=" * 60)


# =============================================================================
# Server Banner
# =============================================================================

def print_banner(host: str, port: int, logger: logging.Logger):
    """Print startup banner with server information."""
    banner = f"""
\033[96m╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║     █████╗ ██╗   ██╗ █████╗     ███████╗███████╗██████╗ ██╗   ██╗███████╗██████╗     ║
║    ██╔══██╗██║   ██║██╔══██╗    ██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗    ║
║    ███████║██║   ██║███████║    ███████╗█████╗  ██████╔╝██║   ██║█████╗  ██████╔╝    ║
║    ██╔══██║╚██╗ ██╔╝██╔══██║    ╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══╝  ██╔══██╗    ║
║    ██║  ██║ ╚████╔╝ ██║  ██║    ███████║███████╗██║  ██║ ╚████╔╝ ███████╗██║  ██║    ║
║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝    ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝    ║
║                                                                                      ║
║\033[0m\033[97m                    Cortex-Medulla Architecture v{__version__:<30}\033[96m║
╚══════════════════════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)

    # System info
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("")
    logger.info(f"\033[93mServer:\033[0m    http://{host}:{port}")
    logger.info(f"\033[93mWebSocket:\033[0m ws://{host}:{port}/ws")
    logger.info("")
    logger.info("\033[96mEndpoints:\033[0m")
    logger.info("  \033[92mGET\033[0m   /health    - Health check")
    logger.info("  \033[92mGET\033[0m   /status    - System status")
    logger.info("  \033[92mGET\033[0m   /system    - Detailed diagnostics")
    logger.info("  \033[93mPOST\033[0m  /chat      - Send message")
    logger.info("  \033[93mPOST\033[0m  /think     - Force deep thinking")
    logger.info("  \033[92mGET\033[0m   /tools     - List available tools")
    logger.info("  \033[95mWS\033[0m    /ws        - WebSocket streaming")
    logger.info("")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AVA Server - Standalone Backend for AVA Neural Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Start with defaults (localhost:8085)
  %(prog)s --port 8080              Use custom port
  %(prog)s --host 0.0.0.0           Allow remote connections
  %(prog)s --check                  Run checks and exit
  %(prog)s --verbose --log-file a.log  Debug mode with file logging

Environment:
  OLLAMA_HOST                       Ollama endpoint (default: http://localhost:11434)
  AVA_LOG_LEVEL                     Logging level (default: INFO)
  AVA_DATA_DIR                      Data directory (default: ./data)
"""
    )

    parser.add_argument(
        "--host",
        default=os.environ.get("AVA_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("AVA_PORT", "8085")),
        help="Port to listen on (default: 8085)"
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API endpoint"
    )
    parser.add_argument(
        "--config", "-c",
        default=os.environ.get("AVA_CONFIG"),
        help="Configuration file path"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("AVA_DATA_DIR"),
        help="Data directory for persistence"
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AVA_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Log to file in addition to console"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run preflight checks and exit"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser.parse_args()


async def main():
    """Main entry point for AVA Server."""
    args = parse_args()

    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        verbose=args.verbose
    )

    # Run preflight checks
    logger.info(f"Starting {__app_name__} v{__version__}")
    report = await run_preflight_checks(
        host=args.host,
        port=args.port,
        ollama_host=args.ollama_host,
        config_path=args.config,
        data_dir=args.data_dir
    )

    print_preflight_report(report, logger)

    # Exit if only checking
    if args.check:
        sys.exit(0 if report.all_passed else 1)

    # Exit if critical checks failed
    if report.critical_failed:
        logger.error("Cannot start server due to critical failures.")
        logger.error("Fix the issues above and try again.")
        sys.exit(1)

    # Check for aiohttp
    if not AIOHTTP_AVAILABLE:
        logger.error("aiohttp is required to run the server.")
        logger.error("Install with: pip install aiohttp")
        sys.exit(1)

    # Import and run the main server
    # We import here to avoid import errors if dependencies are missing
    try:
        # Import the existing server module
        from server import create_app

        app = create_app()

        print_banner(args.host, args.port, logger)

        # Setup graceful shutdown
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Shutdown signal received...")
            shutdown_event.set()

        # Register signal handlers
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)

        # Run the server
        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, args.host, args.port)
        await site.start()

        logger.info("Server ready. Press Ctrl+C to stop.")

        # Wait for shutdown
        try:
            if sys.platform == "win32":
                # Windows doesn't support add_signal_handler, use keyboard interrupt
                await asyncio.get_event_loop().run_in_executor(None, lambda: None)
                while not shutdown_event.is_set():
                    await asyncio.sleep(1)
            else:
                await shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        # Cleanup
        logger.info("Shutting down...")
        await runner.cleanup()
        logger.info("Server stopped.")

    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        logger.error("Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


def run():
    """Entry point for the application."""
    if sys.platform == "win32":
        # Windows requires this for proper async handling
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
