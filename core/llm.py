"""
LLM Integration — Google Gemini client.

Hardening over the previous version:
  * a hard per-request timeout (a hung call used to stall the voice loop
    forever, with the caller hearing silence)
  * retries on 429 AND on 500/502/503/504 and transport errors, with
    exponential backoff and jitter — previously only the literal string "429"
    was matched, so a transient 503 propagated and killed the process
  * `generate()` never raises. It returns a spoken fallback and an `ok` flag,
    so one bad API call degrades a single turn instead of ending the call
  * thinking disabled by default on gemini-2.5-* — it adds latency and
    consumes max_output_tokens, which can leave zero tokens for the answer
  * a grounding contract in the system prompt: state only what is in the
    knowledge block, otherwise offer to check
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from config import cfg
from core.nlu.entities import entity_store

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12

# Errors worth retrying — transient server/transport problems.
_RETRYABLE_CODES = (429, 500, 502, 503, 504)
_RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504", "rate limit", "resource_exhausted",
    "unavailable", "deadline", "timeout", "timed out", "connection",
    "temporarily", "overloaded", "internal error",
)

_FALLBACK_RESPONSE = (
    "Sorry, I'm having trouble looking that up right now. "
    "Could you try asking again, or would you like me to put you through to a colleague?"
)


def build_system_prompt() -> str:
    """Compose the system prompt from config — one source of truth for facts."""
    b = cfg.business
    return f"""You are Sarah, the AI receptionist for {b.name}.

CLINIC KNOWLEDGE (the ONLY facts you may state):
- Doctors: {entity_store.doctor_roster_text()}
- Opening hours: {b.hours_human}. Closed outside those hours.
- Address: {b.address}
- Services: {', '.join(entity_store.service_names())}

GROUNDING CONTRACT — this is your most important rule:
- You may ONLY state a fact that appears in CLINIC KNOWLEDGE above or in the
  CONVERSATION CONTEXT supplied below.
- If you are asked anything else (parking, insurance, wait times, prices,
  specific availability, medical advice), you MUST NOT guess. Say you don't
  have that information and offer to check with a colleague.
- Never invent doctors, services, times, prices or policies.

STYLE:
- This is a phone call. Keep replies to 1-2 short sentences.
- Warm, calm and professional. Never use markdown, bullets or emoji.
- Never give medical advice. If someone describes an emergency, tell them to
  hang up and dial 911 immediately.
"""


def shape_for_speech(text: str) -> str:
    """Clean LLM output for natural speech synthesis."""
    if not text:
        return ""
    s = text
    s = re.sub(r"```[\s\S]*?```", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"https?://\S+", "", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\be\.g\.\s*", "for example ", s)
    s = re.sub(r"\bi\.e\.\s*", "that is ", s)
    s = s.replace(" etc.", " and so on")
    s = re.sub(r"\n{2,}", ". ", s)
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s.strip()


@dataclass
class LLMResponse:
    """Outcome of a generation attempt."""

    text: str
    elapsed: float
    ok: bool = True
    error: Optional[str] = None

    # Legacy call sites unpack `text, elapsed = llm.generate(...)`.
    def __iter__(self):
        return iter((self.text, self.elapsed))


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RETRYABLE_MARKERS)


class LLM:
    """Google Gemini client for conversational responses."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        # `or` would let an explicit empty string fall back to config,
        # making the unconfigured path untestable.
        self.api_key = api_key if api_key is not None else cfg.llm.api_key
        self.model_name = model or cfg.llm.model
        self.max_tokens = max_tokens or cfg.llm.max_tokens
        self.temperature = temperature if temperature is not None else cfg.llm.temperature
        self.system_prompt = system_prompt or build_system_prompt()

        self._client = None
        self._history: list[dict[str, Any]] = []

    # ── client ──────────────────────────────────────────────
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your-api-key-here"

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai
        from google.genai import types as genai_types

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=genai_types.HttpOptions(timeout=int(cfg.llm.timeout_s * 1000)),
        )
        logger.info(
            f"Gemini client ready (model={self.model_name}, timeout={cfg.llm.timeout_s}s)"
        )

    def _build_config(self, sys_prompt: str):
        from google.genai import types

        kwargs: dict[str, Any] = dict(
            system_instruction=sys_prompt,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # gemini-2.5-* think by default; that latency and token spend is not
        # acceptable on a phone call, and it can consume the entire output
        # budget leaving an empty response.
        if "2.5" in self.model_name and cfg.llm.thinking_budget is not None:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=cfg.llm.thinking_budget
                )
            except Exception as e:
                logger.debug(f"ThinkingConfig unsupported for {self.model_name}: {e}")
        return types.GenerateContentConfig(**kwargs)

    def _build_contents(
        self, user_message: str, context: Optional[dict] = None
    ) -> tuple[list[dict], str]:
        contents = list(self._history)
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        prompt = self.system_prompt
        if context:
            lines = []
            slots = {k: v for k, v in (context.get("slots") or {}).items() if v}
            if slots:
                lines.append(
                    "Details already collected: "
                    + ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in slots.items())
                )
            if context.get("active_intent"):
                lines.append(f"Caller is currently trying to: {context['active_intent']}")
            if lines:
                prompt += (
                    "\n\nCONVERSATION CONTEXT:\n"
                    + "\n".join(f"- {ln}" for ln in lines)
                    + "\nDo not ask again for details already collected."
                )
        return contents, prompt

    # ── generation ──────────────────────────────────────────
    def generate(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        shape: bool = True,
        context: Optional[dict] = None,
    ) -> LLMResponse:
        """Generate a response. Never raises — returns a spoken fallback instead."""
        t0 = time.time()

        if not self.is_configured:
            logger.error("LLM called without GEMINI_API_KEY")
            return LLMResponse(_FALLBACK_RESPONSE, time.time() - t0, ok=False,
                               error="missing_api_key")

        try:
            self._ensure_client()
        except Exception as e:
            logger.error(f"LLM client init failed: {e}")
            return LLMResponse(_FALLBACK_RESPONSE, time.time() - t0, ok=False, error=str(e))

        contents, dynamic_prompt = self._build_contents(message, context)
        sys_prompt = system_prompt or dynamic_prompt

        response = None
        last_error: Optional[Exception] = None
        for attempt in range(1, cfg.llm.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._build_config(sys_prompt),
                )
                break
            except Exception as e:
                last_error = e
                if attempt >= cfg.llm.max_retries or not _is_retryable(e):
                    logger.error(f"LLM call failed ({type(e).__name__}): {e}")
                    break
                delay = cfg.llm.retry_base_delay_s * (2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.25)  # jitter
                logger.warning(
                    f"LLM attempt {attempt}/{cfg.llm.max_retries} failed "
                    f"({type(e).__name__}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        elapsed = time.time() - t0

        if response is None:
            return LLMResponse(
                _FALLBACK_RESPONSE, elapsed, ok=False,
                error=str(last_error) if last_error else "no_response",
            )

        # Accessing .text raises when the candidate was safety-blocked or has
        # no parts, so it is guarded rather than trusted.
        try:
            raw_text = response.text or ""
        except (ValueError, AttributeError, IndexError):
            raw_text = ""

        if not raw_text.strip():
            finish = None
            try:
                finish = response.candidates[0].finish_reason
            except Exception:
                pass
            logger.warning(f"LLM returned no usable text (finish_reason={finish})")
            return LLMResponse(
                "Sorry, I didn't quite catch that. Could you say it another way?",
                elapsed, ok=False, error=f"empty_response:{finish}",
            )

        self._history.append({"role": "user", "parts": [{"text": message}]})
        self._history.append({"role": "model", "parts": [{"text": raw_text}]})
        self._trim_history()

        text = shape_for_speech(raw_text) if shape else raw_text
        logger.info(f"LLM ({elapsed:.2f}s): {message[:40]!r} -> {text[:60]!r}")
        return LLMResponse(text, elapsed, ok=True)

    def generate_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> Iterator[str]:
        """Stream response chunks. Synchronous generator — the caller decides
        whether to run it off the main thread.

        (This was previously an `async def` that iterated a synchronous
        iterator, which would have blocked the event loop; nothing called it.)
        """
        if not self.is_configured:
            yield _FALLBACK_RESPONSE
            return
        try:
            self._ensure_client()
            contents, dynamic_prompt = self._build_contents(message, context)
            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=self._build_config(system_prompt or dynamic_prompt),
            )
            full = ""
            for chunk in stream:
                piece = getattr(chunk, "text", None)
                if piece:
                    full += piece
                    yield piece
            if full:
                self._history.append({"role": "user", "parts": [{"text": message}]})
                self._history.append({"role": "model", "parts": [{"text": full}]})
                self._trim_history()
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            yield _FALLBACK_RESPONSE

    # ── history ─────────────────────────────────────────────
    def _trim_history(self) -> None:
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]

    def clear_history(self) -> None:
        self._history.clear()
        logger.debug("Conversation history cleared")

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def is_initialized(self) -> bool:
        return self._client is not None
