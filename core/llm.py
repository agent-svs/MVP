"""
LLM Integration — Google Gemini API Client

Provides:
- Gemini client with env-configurable model, key, temp, max tokens
- Prompt composer (system prompt + conversation history + user query)
- Streaming token support (async)
- Response shaper: strips markdown formatting for natural speech output

Configurable via .env:
  - GEMINI_API_KEY: your API key
  - GEMINI_MODEL: model name (gemini-2.0-flash, gemini-2.5-pro, etc.)
  - GEMINI_MAX_TOKENS: max output tokens
  - GEMINI_TEMPERATURE: creativity control (0.0–2.0)
"""

import logging
import re
import time
from typing import AsyncIterator, Optional

from config import cfg

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """You are a professional, friendly AI voice receptionist for an office.

Your responsibilities:
- Greet callers warmly
- Help with appointment booking, cancellation, and rescheduling
- Answer questions about services and hours
- Escalate to a human when needed

Guidelines:
- Keep responses SHORT (1-3 sentences) — this is a voice conversation
- Be conversational, not robotic
- Never use markdown formatting (no bold, bullets, headers)
- Use natural speech patterns appropriate for being spoken aloud
- If you don't know something, say so honestly
- Confirm details before taking action"""


# ── Response Shaper ────────────────────────────────────────
def shape_for_speech(text: str) -> str:
    """
    Clean LLM output for natural speech synthesis.

    Strips markdown formatting, code blocks, URLs, and other
    text artifacts that sound wrong when read aloud.
    """
    if not text:
        return ""

    s = text

    # Remove code blocks
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Remove markdown headers
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)

    # Remove bold/italic
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", s)

    # Remove markdown links: [text](url) → text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Remove bullet points and numbered lists
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)

    # Remove URLs
    s = re.sub(r"https?://\S+", "", s)

    # Expand common abbreviations for speech
    s = s.replace("&", "and")
    s = re.sub(r"\be\.g\.\s*", "for example ", s)
    s = re.sub(r"\bi\.e\.\s*", "that is ", s)
    s = s.replace(" etc.", " and so on")

    # Clean up whitespace
    s = re.sub(r"\n{2,}", ". ", s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip()

    # Remove trailing periods if duplicated
    s = re.sub(r"\.{2,}", ".", s)

    return s


# ── LLM Client ─────────────────────────────────────────────
class LLM:
    """
    Google Gemini API client for conversational AI.

    Features:
    - Configurable model, temperature, and max tokens
    - Conversation history management
    - Streaming response support
    - Response shaping for speech output
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = api_key or cfg.llm.api_key
        self.model_name = model or cfg.llm.model
        self.max_tokens = max_tokens or cfg.llm.max_tokens
        self.temperature = temperature if temperature is not None else cfg.llm.temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._client = None
        self._history: list[dict[str, str]] = []

    def _ensure_client(self) -> None:
        """Initialize the Gemini client if not already done."""
        if self._client is not None:
            return

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file."
            )

        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        logger.info(f"Gemini client initialized (model={self.model_name})")

    def _build_contents(self, user_message: str, context: Optional[dict] = None) -> tuple[list[dict], str]:
        """Build the message list including history and dynamically updated system prompt."""
        contents = []

        # Add conversation history
        for msg in self._history:
            contents.append(msg)

        # Add current user message
        contents.append({"role": "user", "parts": [{"text": user_message}]})
        
        # Build dynamic system prompt with context
        dynamic_prompt = self.system_prompt
        if context:
            import json
            context_str = json.dumps(context, indent=2)
            dynamic_prompt += f"\n\nCURRENT CONVERSATION CONTEXT (Slots filled so far):\n{context_str}\n\nUse this context to inform your response. If a slot is filled, do not ask for it again."

        return contents, dynamic_prompt

    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        shape: bool = True,
        context: Optional[dict] = None,
    ) -> tuple[str, float]:
        """
        Generate a response from Gemini.

        Args:
            message: User message text
            system_prompt: Override system prompt for this call
            shape: Whether to apply speech shaping
            context: Optional dictionary of conversation state (e.g. active slots)

        Returns:
            Tuple of (response_text, elapsed_seconds)
        """
        self._ensure_client()

        from google.genai import types

        contents, dynamic_prompt = self._build_contents(message, context)
        sys_prompt = system_prompt or dynamic_prompt

        t0 = time.time()

        # Retry with backoff for rate limits (429)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                    logger.warning(f"Rate limited, retrying in {wait}s...")
                    print(f"  ⏳ Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        elapsed = time.time() - t0
        raw_text = response.text or ""

        # Update history
        self._history.append(
            {"role": "user", "parts": [{"text": message}]}
        )
        self._history.append(
            {"role": "model", "parts": [{"text": raw_text}]}
        )

        # Shape for speech
        text = shape_for_speech(raw_text) if shape else raw_text

        logger.info(
            f"LLM ({elapsed:.2f}s): \"{message[:40]}\" → \"{text[:60]}\""
        )
        return text, elapsed

    async def generate_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from Gemini.

        Yields shaped text chunks as they arrive.
        """
        self._ensure_client()

        from google.genai import types

        contents, dynamic_prompt = self._build_contents(message, context)
        sys_prompt = system_prompt or dynamic_prompt

        response = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        full_text = ""
        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text

        # Update history with full response
        self._history.append(
            {"role": "user", "parts": [{"text": message}]}
        )
        self._history.append(
            {"role": "model", "parts": [{"text": full_text}]}
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        logger.debug("Conversation history cleared")

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def is_initialized(self) -> bool:
        return self._client is not None


# ── Standalone test (interactive REPL) ──────────────────────
if __name__ == "__main__":
    cfg.setup_logging()
    print("💬 Gemini LLM — Interactive Test")
    print(f"   Model: {cfg.llm.model}")
    print(f"   Temp:  {cfg.llm.temperature}")
    print(f"   Max tokens: {cfg.llm.max_tokens}")
    print(f"   Type 'quit' to exit, 'clear' to reset history")
    print()

    llm = LLM()
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            llm.clear_history()
            turn = 0
            print("  [History cleared]\n")
            continue

        turn += 1
        try:
            response, elapsed = llm.generate(user_input)
            print(f"AI ({elapsed:.2f}s): {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
