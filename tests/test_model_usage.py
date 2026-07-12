from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.analysis.utils import llm_utils


class FakeConfig:
    def __init__(self, *, retries=1, streaming=False):
        self.retries = retries
        self.streaming = streaming

    def get_llm_retries(self):
        return self.retries

    def get_llm_backoff(self):
        return 0

    def get_enable_streaming_llm_call(self):
        return self.streaming

    def get_llm_provider_id(self):
        return "provider"


class FakeContext:
    def __init__(self, results):
        self.llm_generate = AsyncMock(side_effect=results)
        self.provider = SimpleNamespace(
            provider_config={"id": "provider"},
            model_name="gemma-4-31b-it",
        )

    def get_provider_by_id(self, provider_id=None):
        return self.provider if provider_id == "provider" else None


@pytest.fixture(autouse=True)
def reset_breakers():
    llm_utils._circuit_breakers.clear()


@pytest.mark.asyncio
async def test_non_streaming_success_records_once(monkeypatch):
    response = SimpleNamespace(completion_text="ok", usage=object())
    context = FakeContext([response])
    calls = []
    monkeypatch.setattr(llm_utils, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    result = await llm_utils.call_provider_with_retry(
        context, FakeConfig(), "prompt", umo="umo:test", provider_id="provider"
    )

    assert result is response
    assert context.llm_generate.await_count == 1
    assert len(calls) == 1
    assert calls[0]["status"] == "completed"
    assert calls[0]["provider_id"] == "provider"
    assert calls[0]["response"] is response


@pytest.mark.asyncio
async def test_schema_fallback_records_error_then_completed(monkeypatch):
    response = SimpleNamespace(completion_text="ok", usage=object())
    context = FakeContext([RuntimeError("response_format not supported"), response])
    calls = []
    monkeypatch.setattr(llm_utils, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    result = await llm_utils.call_provider_with_retry(
        context,
        FakeConfig(),
        "prompt",
        umo="umo:test",
        provider_id="provider",
        response_format={"type": "json_schema"},
    )

    assert result is response
    assert context.llm_generate.await_count == 2
    assert [call["status"] for call in calls] == ["error", "completed"]


@pytest.mark.asyncio
async def test_open_circuit_records_nothing(monkeypatch):
    context = FakeContext([])
    calls = []
    breaker = SimpleNamespace(
        allow_request=lambda: False,
        record_success=lambda: None,
        record_failure=lambda: None,
    )
    monkeypatch.setattr(llm_utils, "_get_circuit_breaker", lambda _pid: breaker)
    monkeypatch.setattr(llm_utils, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    result = await llm_utils.call_provider_with_retry(
        context, FakeConfig(), "prompt", provider_id="provider"
    )

    assert result is None
    assert context.llm_generate.await_count == 0
    assert calls == []


@pytest.mark.asyncio
async def test_streaming_records_only_final_response(monkeypatch):
    chunks = [
        SimpleNamespace(is_chunk=True, completion_text="a", usage=None),
        SimpleNamespace(is_chunk=False, completion_text="done", usage=object()),
    ]

    class StreamingProvider:
        provider_config = {"id": "provider"}
        model_name = "model"

        async def text_chat_stream(self, **_kwargs):
            for chunk in chunks:
                yield chunk

    context = FakeContext([])
    context.provider = StreamingProvider()
    calls = []
    monkeypatch.setattr(llm_utils, "schedule_model_usage", lambda **kwargs: calls.append(kwargs))

    result = await llm_utils.call_provider_with_retry(
        context, FakeConfig(streaming=True), "prompt", provider_id="provider"
    )

    assert result is chunks[-1]
    assert len(calls) == 1
    assert calls[0]["response"] is chunks[-1]
