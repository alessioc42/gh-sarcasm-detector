from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ollama import Client as OllamaSdkClient
from ollama import ResponseError

logger = logging.getLogger(__name__)

CHAT_TIMEOUT_SECONDS = 600.0
PULL_TIMEOUT_SECONDS = 86400.0

# Expected reply JSON only, e.g. {"sarcastic": false, "confidence": 10}
# (~37 chars, ~15 tokens once tokenized). 20x headroom for fences or brief preamble.
_ESTIMATED_RESPONSE_JSON_TOKENS = 15
MAX_GENERATED_TOKENS = _ESTIMATED_RESPONSE_JSON_TOKENS * 40


@dataclass
class ChatResult:
    raw_body: str
    http_status: int
    duration_ms: int
    request_payload: dict[str, Any]
    error_message: str | None = None


class OllamaClient:
    def __init__(self, endpoint: str, api_token: str | None = None) -> None:
        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = OllamaSdkClient(
            host=endpoint.rstrip("/"),
            headers=headers or None,
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        self._pull_client = OllamaSdkClient(
            host=endpoint.rstrip("/"),
            headers=headers or None,
            timeout=PULL_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self._client.close()
        self._pull_client.close()

    def show_model(self, model: str) -> dict[str, Any]:
        return self._client.show(model).model_dump()

    def model_is_available(self, model: str) -> bool:
        try:
            self._client.show(model)
            return True
        except (ResponseError, ConnectionError):
            return False

    def pull_model(self, model: str) -> None:
        logger.info("Pulling model %s...", model)
        last_status: str | None = None
        for progress in self._pull_client.pull(model, stream=True):
            status = progress.status
            if status and status != last_status:
                logger.info("Pull %s: %s", model, status)
                last_status = status
        logger.info("Model %s pulled successfully", model)

    def delete_model(self, model: str) -> None:
        logger.info("Deleting model %s from Ollama...", model)
        try:
            self._client.delete(model)
            logger.info("Model %s deleted", model)
        except ResponseError as exc:
            if exc.status_code == 404:
                logger.warning(
                    "Model %s not found during delete (already removed?)",
                    model,
                )
                return
            raise

    def model_supports_audio(self, model: str) -> tuple[bool, list[str]]:
        try:
            data = self.show_model(model)
        except ResponseError as exc:
            logger.warning("Could not fetch capabilities for %s: %s", model, exc)
            return False, []

        capabilities = data.get("capabilities") or []
        if isinstance(capabilities, list):
            caps = [str(c).lower() for c in capabilities]
            return "audio" in caps, caps

        return False, []

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str,
        audio_b64: str | None = None,
    ) -> ChatResult:
        user_msg: dict[str, Any] = {"role": "user", "content": user_message}
        if audio_b64:
            user_msg["images"] = [audio_b64]

        messages = [
            {"role": "system", "content": system_prompt},
            user_msg,
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": MAX_GENERATED_TOKENS},
        }

        start = time.monotonic()
        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={"num_predict": MAX_GENERATED_TOKENS},
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            raw = json.dumps(response.model_dump())
            return ChatResult(
                raw_body=raw,
                http_status=200,
                duration_ms=duration_ms,
                request_payload=_redact_payload(payload),
            )
        except ResponseError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            error_text = str(exc.error or exc)
            return ChatResult(
                raw_body=error_text,
                http_status=exc.status_code,
                duration_ms=duration_ms,
                request_payload=_redact_payload(payload),
                error_message=f"HTTP {exc.status_code}: {error_text[:500]}",
            )
        except ConnectionError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ChatResult(
                raw_body="",
                http_status=0,
                duration_ms=duration_ms,
                request_payload=_redact_payload(payload),
                error_message=str(exc),
            )


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Store request metadata without huge base64 blobs."""
    copy = json.loads(json.dumps(payload))
    for msg in copy.get("messages", []):
        if "images" in msg and msg["images"]:
            msg["images"] = [f"<base64:{len(msg['images'][0])} chars>"]
    return copy
