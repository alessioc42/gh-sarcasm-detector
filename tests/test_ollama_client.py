from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest
from ollama import ResponseError

from sarcasm_detector.ollama_client import (
    ChatResult,
    MAX_GENERATED_TOKENS,
    OllamaClient,
    _redact_payload,
)


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


class TestMaxGeneratedTokens:
    def test_limit_is_forty_times_json_estimate(self) -> None:
        assert MAX_GENERATED_TOKENS == 15 * 40


class TestOllamaClient:
    def _client_with_mock(self) -> tuple[OllamaClient, mock.Mock, mock.Mock]:
        client = OllamaClient("http://test")
        sdk = mock.Mock()
        pull_sdk = mock.Mock()
        client._client = sdk
        client._pull_client = pull_sdk
        return client, sdk, pull_sdk

    def test_show_model(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.return_value = SimpleNamespace(
            model_dump=lambda: {"capabilities": ["audio", "completion"]}
        )
        data = client.show_model("test-model")
        assert "audio" in data["capabilities"]
        client.close()

    def test_model_supports_audio_true(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.return_value = SimpleNamespace(
            model_dump=lambda: {"capabilities": ["audio"]}
        )
        supports, caps = client.model_supports_audio("m")
        assert supports is True
        assert caps == ["audio"]
        client.close()

    def test_model_supports_audio_http_error(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.side_effect = ResponseError("error", 500)
        supports, caps = client.model_supports_audio("m")
        assert supports is False
        assert caps == []
        client.close()

    def test_model_supports_audio_invalid_capabilities(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.return_value = SimpleNamespace(
            model_dump=lambda: {"capabilities": "nope"}
        )
        supports, caps = client.model_supports_audio("m")
        assert supports is False
        assert caps == []
        client.close()

    def test_chat_success(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.chat.return_value = SimpleNamespace(
            model_dump=lambda: {
                "message": {"content": '{"sarcastic": true, "confidence": 0.9}'}
            }
        )
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
        sdk.chat.assert_called_once()
        messages = sdk.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["images"] == ["abc"]
        assert sdk.chat.call_args.kwargs["think"] is False
        assert sdk.chat.call_args.kwargs["options"] == {
            "num_predict": MAX_GENERATED_TOKENS
        }
        client.close()

    def test_chat_http_error(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.chat.side_effect = ResponseError("bad request", 400)
        result = client.chat(model="m", system_prompt="s", user_message="u")
        assert result.error_message is not None
        assert result.http_status == 400
        client.close()

    def test_chat_transport_error(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.chat.side_effect = ConnectionError("down")
        result = client.chat(model="m", system_prompt="s", user_message="u")
        assert result.http_status == 0
        assert "down" in result.error_message
        client.close()

    def test_auth_header(self) -> None:
        with mock.patch("sarcasm_detector.ollama_client.OllamaSdkClient") as mock_cls:
            OllamaClient("http://test", api_token="tok123")
            kwargs = mock_cls.call_args.kwargs
            assert kwargs["headers"]["Authorization"] == "Bearer tok123"

    def test_pull_model(self) -> None:
        client, _, pull_sdk = self._client_with_mock()
        pull_sdk.pull.return_value = [
            SimpleNamespace(status="pulling manifest"),
            SimpleNamespace(status="success"),
        ]
        client.pull_model("llama3.2")
        pull_sdk.pull.assert_called_once_with("llama3.2", stream=True)
        client.close()

    def test_delete_model(self) -> None:
        client, sdk, _ = self._client_with_mock()
        client.delete_model("llama3.2")
        sdk.delete.assert_called_once_with("llama3.2")
        client.close()

    def test_delete_model_not_found(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.delete.side_effect = ResponseError("missing", 404)
        client.delete_model("missing")
        client.close()

    def test_list_installed_models(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.list.return_value = SimpleNamespace(
            model_dump=lambda: {
                "models": [
                    {
                        "name": "llama3.2:latest",
                        "size": 1234567890,
                        "modified_at": "2024-01-01T00:00:00Z",
                    },
                ]
            }
        )
        installed = client.list_installed_models()
        assert len(installed) == 1
        assert installed[0].name == "llama3.2:latest"
        assert installed[0].size_bytes == 1234567890
        assert installed[0].modified_at == "2024-01-01T00:00:00Z"
        client.close()

    def test_installed_model_size(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.list.return_value = SimpleNamespace(
            model_dump=lambda: {
                "models": [{"name": "m1", "size": 42, "modified_at": None}]
            }
        )
        assert client.installed_model_size("m1") == 42
        assert client.installed_model_size("missing") is None
        client.close()

    def test_model_is_available(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.return_value = SimpleNamespace(model_dump=lambda: {"capabilities": []})
        assert client.model_is_available("m") is True
        client.close()

    def test_model_is_available_false(self) -> None:
        client, sdk, _ = self._client_with_mock()
        sdk.show.side_effect = ResponseError("missing", 404)
        assert client.model_is_available("m") is False
        client.close()
