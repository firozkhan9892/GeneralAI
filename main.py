"""Application entry point for GeneralAI.

Bootstraps configuration, logging, and launches the application
lifecycle.  Run from the project root::

    python main.py

Use ``--help`` to see available CLI options.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from app import __version__, __app_name__
from app.config.settings import AppSettings
from app.utils.logging import LoggingManager


# ------------------------------------------------------------------
# Banner (printed — not logged — per project rules)
# ------------------------------------------------------------------
BANNER = rf"""
  ____       _           _    _          ___ ___
 / ___|     | |         | |  (_)_       |_ _|_ _|
| |  _  __ _| |__   __ _| | ___| |_ ___  | | | |
| |_| |/ _` | '_ \ / _` | |/ / | __/ _ \ | | | |
|  _  | (_| | |_) | (_| |   <| | ||  __/_| |_| |
|_| \_|\__,_|_.__/ \__,_|_|\_\_|\__\___|___|___|

  {__app_name__} v{__version__}
  Autonomous AI Platform
  https://github.com/anomalyco/GeneralAI
"""  # noqa: W605 (valid escape for banner)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog=__app_name__.lower(),
        description="Autonomous AI Platform",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__app_name__} v{__version__}",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Runtime environment (development, staging, production)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single agent request with this prompt and exit",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="cli",
        help="Session ID for --prompt runs (default: cli)",
    )
    return parser


def _parse_cli(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse CLI arguments with consistent error handling.

    Args:
        parser: The argument parser instance.

    Returns:
        Parsed namespace with known arguments.
    """
    try:
        return parser.parse_args()
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        print(f"ERROR: Failed to parse arguments: {exc}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------
def bootstrap() -> tuple[AppSettings, argparse.Namespace]:
    """Initialise the application: config → logging → directories.

    Returns:
        Tuple of fully resolved :class:`AppSettings` and parsed CLI args.
    """
    # 1. Load settings (reads .env + env vars + defaults)
    settings = AppSettings()

    # 2. Apply CLI overrides
    parser = _build_parser()
    cli_args = _parse_cli(parser)

    if cli_args.env is not None:
        settings.environment = cli_args.env
    if cli_args.log_level is not None:
        settings.log_level = cli_args.log_level.upper()
    if cli_args.debug is True:
        settings.debug = True

    # 3. Ensure required directories exist
    settings.ensure_directories()

    # 4. Bootstrap logging
    manager = LoggingManager(settings)
    manager.bootstrap()

    return settings, cli_args


# ------------------------------------------------------------------
# Application lifecycle
# ------------------------------------------------------------------
class Application:
    """Top-level application orchestrator.

    Owns configuration, logging, and the main run loop.  Future
    modules (brain, memory, agents, …) will be wired in here.
    """

    def __init__(self, settings: AppSettings) -> None:
        """Initialise the Application.

        Args:
            settings: Resolved application settings.
        """
        self._settings = settings
        self._log = LoggingManager(settings).get_logger(__name__)
        self._running = False

    @property
    def settings(self) -> AppSettings:
        """Expose the application settings."""
        return self._settings

    def run(self) -> None:
        """Start the application main loop.

        Currently prints a banner and sits idle.  Future iterations
        will wire in the brain, memory, planners, agents, etc.
        """
        self._running = True
        print(BANNER)
        self._log.info(
            "Application started (environment=%s)", self._settings.environment
        )
        self._log.debug("Settings: %s", self._settings.model_dump())

        try:
            self._idle()
        except KeyboardInterrupt:
            self._log.info("Received shutdown signal (SIGINT)")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shut down the application."""
        if not self._running:
            return
        self._running = False
        self._log.info("Application shutdown complete")

    def _idle(self) -> None:
        """Block until interrupted.

        Override this method in subclasses to inject custom behaviour
        without changing the public interface.
        """
        import time

        while self._running:
            time.sleep(1)


# ------------------------------------------------------------------
# Agent CLI
# ------------------------------------------------------------------
async def run_agent_cli(prompt: str, session_id: str) -> int:
    """Boot the kernel and execute a single agent request.

    Registers the built-in tool catalogue so the agent loop has real
    tools (including the ``echo`` fallback) available.

    Args:
        prompt: Raw user input to process.
        session_id: Session identifier for the run.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    from app.core.container import DependencyContainer
    from app.core.lifecycle import LifecycleManager
    from app.kernel import AgentRequest, AgentRunConfig
    from app.kernel.bootstrap import bootstrap_kernel
    from app.tools.categories.planning import plan_tools
    from app.tools.registry import ToolRegistry

    container = DependencyContainer()
    lifecycle = LifecycleManager()
    orchestrator = bootstrap_kernel(container, lifecycle)

    registry = container.resolve(ToolRegistry)
    registry.discover()
    for tool in plan_tools():
        registry.register(tool)

    print(BANNER)
    print(f"Running agent with prompt: {prompt!r}\n")

    response = await orchestrator.run_agent(
        AgentRequest(raw_input=prompt, session_id=session_id),
        config=AgentRunConfig(session_id=session_id),
    )

    print(f"[{response.status.value}] {response.output.content}")
    if response.summary.total_steps:
        print(
            f"steps={response.summary.total_steps} "
            f"succeeded={response.summary.succeeded} "
            f"failed={response.summary.failed} "
            f"skipped={response.summary.skipped} "
            f"retries={response.summary.retries} "
            f"tools={','.join(response.summary.tools_invoked) or '-'}"
        )
    if response.error:
        print(f"error: {response.error}")
    return 0 if response.success else 1


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> NoReturn:
    """Application entry point.

    Parses CLI arguments, bootstraps the environment, creates the
    :class:`Application`, and runs it.  When ``--prompt`` is given,
    runs a single agent request instead and exits.  Exits with a
    non-zero code on failure.
    """
    try:
        settings, cli_args = bootstrap()

        if cli_args.prompt is not None:
            import asyncio

            code = asyncio.run(run_agent_cli(cli_args.prompt, cli_args.session_id))
            sys.exit(code)

        app = Application(settings)
        app.run()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
