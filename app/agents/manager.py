"""Agent manager — supervises multiple concurrent agent sessions.

The :class:`AgentManager` is the application-level orchestrator for the
:class:`AgentRuntime`.  It creates and tracks one :class:`AgentSession`
per request, drives each through the runtime in a background task,
supports cancellation, pause/resume, retries and timeouts, and persists
session state through a :class:`SessionRegistry`.

Design notes:
    - ``execute`` starts a run and returns immediately so callers can
      manage several sessions concurrently.
    - ``await_completion`` blocks until a session reaches a terminal
      state (or a paused state after ``pause``).
    - Cancellation is cooperative: the manager hands each session its
      own :class:`CancellationToken`, which the runtime observes between
      steps.
    - Pause is implemented by cancelling the active run and re-launching
      the same request on resume; partial steps are not checkpointed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from app.agents.exceptions import (
    SessionAlreadyExistsError,
    SessionNotRunnableError,
)
from app.agents.models import (
    TERMINAL_SESSION_STATUSES,
    AgentExecutionOptions,
    AgentSession,
    SessionStatus,
)
from app.agents.persistence import SessionStore
from app.agents.registry import SessionRegistry
from app.kernel.agent.models import (
    AgentRequest,
    AgentResponse,
    AgentRunConfig,
    AgentStatus,
)
from app.kernel.agent.runtime import AgentRuntime
from app.kernel.memory.engine import MemoryEngine
from app.kernel.memory.models import MemoryQuery, MemoryRecord
from app.tools.base import Tool
from app.tools.context import CancellationToken
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class AgentManager:
    """Supervises concurrent agent sessions backed by an AgentRuntime.

    Args:
        runtime: Optional :class:`AgentRuntime`; created on demand.
        registry: Optional :class:`SessionRegistry`; created on demand.
        session_store: Optional :class:`SessionStore` used to build a
            registry when ``registry`` is omitted.
        default_options: Default :class:`AgentExecutionOptions` used by
            ``execute`` when no per-call options are supplied.
    """

    def __init__(
        self,
        *,
        runtime: AgentRuntime | None = None,
        registry: SessionRegistry | None = None,
        session_store: SessionStore | None = None,
        default_options: AgentExecutionOptions | None = None,
    ) -> None:
        self._runtime = runtime or AgentRuntime()
        self._registry = registry or SessionRegistry(store=session_store)
        self._default_options = default_options or AgentExecutionOptions()
        self._tasks: dict[str, asyncio.Task] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._paused: set[str] = set()
        self._cancel_reasons: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def runtime(self) -> AgentRuntime:
        """Return the underlying agent runtime."""
        return self._runtime

    @property
    def registry(self) -> SessionRegistry:
        """Return the session registry."""
        return self._registry

    @property
    def memory(self) -> MemoryEngine:
        """Return the runtime's shared memory engine."""
        return self._runtime.memory

    @property
    def tools(self) -> ToolRegistry:
        """Return the runtime's shared tool registry."""
        return self._runtime.registry

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: AgentRequest,
        *,
        config: AgentRunConfig | None = None,
        options: AgentExecutionOptions | None = None,
    ) -> AgentSession:
        """Create a session and start its run as a background task.

        Args:
            request: The agent request.  A session id is generated when
                ``request.session_id`` is empty.
            config: Optional per-run config override.
            options: Optional execution options override.

        Returns:
            The new session (already ``RUNNING``).

        Raises:
            SessionAlreadyExistsError: If a session with the same id
                already exists.
        """
        session_id = request.session_id or uuid.uuid4().hex
        if self._registry.has(session_id):
            raise SessionAlreadyExistsError(
                f"Session '{session_id}' already exists",
                module="agents.manager",
            )

        request = request.model_copy(update={"session_id": session_id})
        cfg = config or request.config
        if cfg is not None and not cfg.session_id:
            cfg = cfg.model_copy(update={"session_id": session_id})
        opts = options or self._default_options

        session = AgentSession(
            session_id=session_id,
            request=request,
            config=cfg,
            options=opts,
            status=SessionStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self._registry.register(session)
        if opts.persist:
            await self._registry.persist(session)

        token = CancellationToken()
        self._tokens[session_id] = token
        task = asyncio.create_task(self._run(session_id))
        self._tasks[session_id] = task
        log.info("Agent session %s started", session_id)
        return session

    async def await_completion(
        self,
        session_id: str,
        timeout_s: float | None = None,
    ) -> AgentSession:
        """Wait until a session's current run task finishes.

        Args:
            session_id: The session to wait on.
            timeout_s: Optional timeout; if exceeded the run continues in
                the background and the current session is returned.

        Returns:
            The session snapshot after the run finished.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        session = self._registry.get_or_raise(session_id)
        task = self._tasks.get(session_id)
        if task is None:
            return session
        if timeout_s is None:
            await task
        else:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
            except asyncio.TimeoutError:
                log.info(
                    "await_completion timed out for %s; run continues",
                    session_id,
                )
        return self._registry.get_or_raise(session_id)

    # ------------------------------------------------------------------
    # Lifecycle control
    # ------------------------------------------------------------------

    async def cancel(
        self, session_id: str, reason: str = "user_requested"
    ) -> AgentSession:
        """Cancel an active or paused session.

        Args:
            session_id: The session to cancel.
            reason: Cancellation reason stored on the session.

        Returns:
            The session snapshot (``CANCELLED``).

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        session = self._registry.get_or_raise(session_id)
        if session.status in TERMINAL_SESSION_STATUSES:
            return session

        if session.status == SessionStatus.PAUSED:
            return await self._finalize(
                session_id, SessionStatus.CANCELLED, error=reason
            )

        self._cancel_reasons[session_id] = reason
        token = self._tokens.get(session_id)
        if token is not None:
            token.cancel()
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            await task
        return self._registry.get_or_raise(session_id)

    async def pause(self, session_id: str) -> AgentSession:
        """Pause a running session.

        The active run is cancelled cooperatively; the session retains
        its request/config so it can be resumed later.

        Args:
            session_id: The session to pause.

        Returns:
            The session snapshot (``PAUSED``).

        Raises:
            SessionNotFoundError: If the session does not exist.
            SessionNotRunnableError: If the session is already terminal
                or already paused.
        """
        session = self._registry.get_or_raise(session_id)
        if session.status in TERMINAL_SESSION_STATUSES:
            raise SessionNotRunnableError(
                f"Cannot pause terminal session '{session_id}' "
                f"(status={session.status.value})",
                module="agents.manager",
            )
        if session.status == SessionStatus.PAUSED:
            raise SessionNotRunnableError(
                f"Session '{session_id}' is already paused",
                module="agents.manager",
            )

        self._paused.add(session_id)
        token = self._tokens.get(session_id)
        if token is not None:
            token.cancel()
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            await task
        return self._registry.get_or_raise(session_id)

    def resume(self, session_id: str) -> AgentSession:
        """Resume a paused session by re-launching its request.

        Args:
            session_id: The session to resume.

        Returns:
            The session snapshot (``RUNNING``).

        Raises:
            SessionNotFoundError: If the session does not exist.
            SessionNotRunnableError: If the session is not paused.
        """
        session = self._registry.get_or_raise(session_id)
        if session.status != SessionStatus.PAUSED:
            raise SessionNotRunnableError(
                f"Cannot resume session '{session_id}' in state "
                f"'{session.status.value}'",
                module="agents.manager",
            )

        self._paused.discard(session_id)
        updated = session.model_copy(
            update={
                "status": SessionStatus.RUNNING,
                "response": None,
                "error": None,
                "completed_at": None,
                "updated_at": datetime.utcnow(),
            }
        )
        self._registry.update(updated)

        token = CancellationToken()
        self._tokens[session_id] = token
        task = asyncio.create_task(self._run(session_id))
        self._tasks[session_id] = task
        log.info("Agent session %s resumed", session_id)
        return updated

    async def delete_session(self, session_id: str) -> bool:
        """Cancel if active, then remove the session from memory and store.

        Args:
            session_id: The session to delete.

        Returns:
            ``True`` if a session was removed.
        """
        session = self._registry.get(session_id)
        if session is not None and not session.is_terminal:
            await self.cancel(session_id)
        return await self._registry.delete(session_id)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_status(self, session_id: str) -> AgentSession:
        """Return the session snapshot for *session_id*.

        Args:
            session_id: The session identifier.

        Returns:
            The session (its ``status`` field describes the lifecycle).

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        return self._registry.get_or_raise(session_id)

    def get_session(self, session_id: str) -> AgentSession | None:
        """Return the session snapshot, or ``None`` if absent."""
        return self._registry.get(session_id)

    def list_sessions(
        self,
        *,
        status: SessionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentSession]:
        """List sessions, optionally filtered by status.

        Args:
            status: Optional status filter.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip (newest first).

        Returns:
            Matching sessions, newest first.
        """
        return self._registry.list(status=status, limit=limit, offset=offset)

    def active_count(self) -> int:
        """Return the number of currently running sessions."""
        return len(self._tasks)

    # ------------------------------------------------------------------
    # Memory and tools
    # ------------------------------------------------------------------

    async def memory_for_session(
        self, session_id: str, *, limit: int = 20
    ) -> list[MemoryRecord]:
        """Return memory records recorded for a session.

        Args:
            session_id: The session identifier.
            limit: Maximum number of records to return.

        Returns:
            Records for the session, newest first.
        """
        return await self._runtime.memory.retrieve(
            MemoryQuery(session_id=session_id, limit=limit)
        )

    def register_tool(self, tool: Tool) -> None:
        """Register a tool on the shared runtime tool registry.

        Args:
            tool: The tool to register.
        """
        self._runtime.registry.register(tool)

    def tool_names(self) -> list[str]:
        """Return the names of tools available to sessions."""
        return self._runtime.registry.names()

    # ------------------------------------------------------------------
    # Persistence / lifecycle
    # ------------------------------------------------------------------

    async def restore(self) -> int:
        """Load persisted sessions into the registry.

        Returns:
            The number of sessions loaded.
        """
        return await self._registry.load_all()

    async def shutdown(self) -> None:
        """Cancel all active sessions and wait for their tasks."""
        for token in self._tokens.values():
            token.cancel()
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._tokens.clear()
        self._paused.clear()
        self._cancel_reasons.clear()
        log.info("Agent manager shut down")

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run(self, session_id: str) -> None:
        """Drive a session to a terminal state, applying retries."""
        while True:
            session = self._registry.get(session_id)
            if session is None:
                return
            options = session.options
            attempt = session.attempts + 1
            session = self._mark_running(session, attempt)

            token = self._tokens.get(session_id)
            if token is None:
                token = CancellationToken()
                self._tokens[session_id] = token

            try:
                response = await self._runtime.run(
                    session.request,
                    config=session.config,
                    cancellation_token=token,
                )
            except asyncio.CancelledError:
                await self._handle_cancelled(session_id, "Agent run task cancelled")
                return
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "Agent session %s raised %s", session_id, type(exc).__name__
                )
                await self._finalize(session_id, SessionStatus.FAILED, error=str(exc))
                return

            if response.status == AgentStatus.SUCCEEDED:
                await self._finalize(
                    session_id, SessionStatus.COMPLETED, response=response
                )
                return
            if response.status == AgentStatus.CANCELLED:
                await self._handle_cancelled(
                    session_id, response.error or "Agent run cancelled"
                )
                return
            if response.status == AgentStatus.TIMED_OUT:
                await self._finalize(
                    session_id,
                    SessionStatus.TIMED_OUT,
                    response=response,
                    error=response.error or "Agent run timed out",
                )
                return

            # FAILED — retry if the attempt budget allows it.
            if attempt < options.max_attempts:
                if options.retry_delay_s > 0:
                    await asyncio.sleep(options.retry_delay_s)
                continue
            await self._finalize(
                session_id,
                SessionStatus.FAILED,
                response=response,
                error=response.error or "Agent run failed",
            )
            return

    def _mark_running(self, session: AgentSession, attempt: int) -> AgentSession:
        """Stamp a session as running for *attempt* and store it."""
        updated = session.model_copy(
            update={
                "status": SessionStatus.RUNNING,
                "attempts": attempt,
                "started_at": session.started_at or datetime.utcnow(),
                "error": None,
                "updated_at": datetime.utcnow(),
            }
        )
        self._registry.update(updated)
        return updated

    async def _handle_cancelled(self, session_id: str, message: str) -> AgentSession:
        """Reconcile a cancelled run: paused or cancelled."""
        if session_id in self._paused:
            self._paused.discard(session_id)
            return await self._finalize(session_id, SessionStatus.PAUSED)
        reason = self._cancel_reasons.pop(session_id, message)
        return await self._finalize(session_id, SessionStatus.CANCELLED, error=reason)

    async def _finalize(
        self,
        session_id: str,
        status: SessionStatus,
        *,
        response: AgentResponse | None = None,
        error: str | None = None,
    ) -> AgentSession:
        """Transition a session into a final state and persist it."""
        session = self._registry.get_or_raise(session_id)
        updated = session.with_status(
            status,
            error=error,
            response=response,
            completed=status in TERMINAL_SESSION_STATUSES,
        )
        self._registry.update(updated)
        self._tasks.pop(session_id, None)
        self._tokens.pop(session_id, None)
        self._paused.discard(session_id)
        self._cancel_reasons.pop(session_id, None)
        if session.options.persist:
            await self._registry.persist(updated)
        log.info("Agent session %s finalized (%s)", session_id, status.value)
        return updated
