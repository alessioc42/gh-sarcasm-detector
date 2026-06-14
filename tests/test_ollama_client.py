from __future__ import annotations

import json
from unittest import mock

import httpx
import pytest

from sarcasm_detector.ollama_client import ChatResult, OllamaClient, _redact_payload


class TestRedactPayload:
    def test_redacts_images(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "hi", "images": ["A" * 100]},
            ]
        }
        redacted = _redact_payload(payload)
        assert redacted["messages"][0]["images"] == ["<base64:100 chars>"]

    def test_no_images_unchanged(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        assert _redact_payload(payload) == payload


class TestOllamaClient:
    def _client_with_handler(self, handler):
        client = OllamaClient("http://test")
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://test",
        )
        return client

    def test_show_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/show"
            return httpx.Response(200, json={"capabilities": ["audio", "completion"]})

        client = self._client_with_handler(handler)
        data = client.show_model("test-model")
        assert "audio" in data["capabilities"]
        client.close()

    def test_model_supports_audio_true(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"capabilities": ["audio"]})

        client = self._client_with_handler(handler)
        supports, caps = client.model_supports_audio("m")
        assert supports is True
        assert caps == ["audio"]
        client.close()

    def test_model_supports_audio_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="error")

        client = self._client_with_handler(handler)
        supports, caps = client.model_supports_audio("m")
        assert supports is False
        assert caps == []
        client.close()

    def test_model_supports_audio_invalid_capabilities(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"capabilities": "nope"})

        client = self._client_with_handler(handler)
        supports, caps = client.model_supports_audio("m")
        assert supports is False
        assert caps == []
        client.close()

    def test_chat_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is False
            assert body["messages"][0]["role"] == "system"
            return httpx.Response(
                200,
                json={"message": {"content": '{"sarcastic": true, "confidence": 0.9}'}},
            )

        client = self._client_with_handler(handler)
        result = client.chat(
            model="m",
            system_prompt="sys",
            user_message="text",
            audio_b64="abc",
        )
        assert isinstance(result, ChatResult)
        assert result.error_message is None
        assert result.http_status == 200
        assert "sarcastic" in result.raw_body
        client.close()

    def test_chat_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad request")

        client = self._client_with_handler(handler)
        result = client.chat(model="m", system_prompt="s", user_message="u")
        assert result.error_message is not None
        assert result.http_status == 400
        client.close()

    def test_chat_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = self._client_with_handler(handler)
        result = client.chat(model="m", system_prompt="s", user_message="u")
        assert result.http_status == 0
        assert "down" in result.error_message
        client.close()

    def test_auth_header(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"capabilities": []})

        client = OllamaClient("http://test", api_token="tok123")
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://test",
            headers={"Authorization": "Bearer tok123"},
        )
        client.model_supports_audio("m")
        assert captured["auth"] == "Bearer tok123"
        client.close()
