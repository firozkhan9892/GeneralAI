"""Cognitive LLM adapter â€” narrow abstraction for LLM generation."""

from __future__ import annotations

from typing import Any

from app.llm.llm_router import LLMRouter
from app.llm.models import ChatRequest, Message, Role


class LLMCognitiveAdapter:
    """Narrow abstraction at the cognitive boundary for LLM generation.

    This adapter delegates generation to the application-level LLMRouter
    without leaking provider-specific details or advanced router logic
    into the deterministic cognitive pipeline.
    """

    def __init__(self, router: LLMRouter) -> None:
        """Initialize the adapter.

        Args:
            router: The application-level LLMRouter instance.
        """
        self._router = router

    async def generate_response(
        self,
        context: dict[str, Any],
        structured_output: str,
    ) -> str:
        """Construct a contextual natural language response via the LLM.

        Args:
            context: The cognitive context dictionary. Expected keys are
                ``user_input``, ``session_id``, and ``intent``.
            structured_output: The deterministic output string to enhance.

        Returns:
            The synthesized natural language response, or the deterministic
            output when the model returns no usable content.

        Raises:
            Exception: Router and provider failures propagate to the caller.
                The adapter does not replace a failed model call with an
                apparently successful LLM response.
        """
        user_input = str(context.get("user_input") or "")
        session_id = str(context.get("session_id") or "unknown")
        intent = str(context.get("intent") or "unknown")
        prompt = (
            "You are an AI assistant orchestrating a cognitive pipeline.\n\n"
            "Use the user request as the primary question and treat the "
            "structured result as authoritative facts from the completed "
            "cognitive work. Please synthesize this into a clear, "
            "contextual, natural-language response for the user.\n\n"
            f"User request:\n{user_input or '(no user input provided)'}\n\n"
            f"Session: {session_id}\n"
            f"Intent: {intent}\n\n"
            f"=== STRUCTURED RESULT ===\n{structured_output}\n"
            "=========================\n\n"
            "Keep the response concise, helpful, and directly related "
            "to the user request and structured output. Do not mention the "
            "'cognitive engines' or internal architecture directly to the user."
        )

        request = ChatRequest(
            messages=(Message(role=Role.USER, content=prompt),),
            model="default",
            stream=False,
        )

        response = await self._router.generate_async(request)
        if (response.content or "").strip():
            return response.content
        return structured_output
