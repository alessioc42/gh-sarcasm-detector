from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    raw_body: str
    http_status: int
    duration_ms: int
    request_payload: dict[str, Any]
    error_message: str | None = None


class OllamaClient:
    def __init__(self, endpoint: str, api_token: str | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = httpx.Client(base_url=self.endpoint, headers=headers, timeout=600.0)

    def close(self) -> None:
        self._client.close()

    def show_model(self, model: str) -> dict[str, Any]:
        resp = self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    def model_supports_audio(self, model: str) -> tuple[bool, list[str]]:
        try:
            data = self.show_model(model)
        except httpx.HTTPError as exc:
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
        message: dict[str, Any] = {"role": "user", "content": user_message}
        if audio_b64:
            message["images"] = [audio_b64]

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                message,
            ],
            "stream": False,
        }

        start = time.monotonic()
        try:
            resp = self._client.post("/api/chat", json=payload)
            duration_ms = int((time.monotonic() - start) * 1000)
            raw = resp.text
            if resp.is_success:
                return ChatResult(
                    raw_body=raw,
                    http_status=resp.status_code,
                    duration_ms=duration_ms,
                    request_payload=_redact_payload(payload),
                )
            return ChatResult(
                raw_body=raw,
                http_status=resp.status_code,
                duration_ms=duration_ms,
                request_payload=_redact_payload(payload),
                error_message=f"HTTP {resp.status_code}: {raw[:500]}",
            )
        except httpx.HTTPError as exc:
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
