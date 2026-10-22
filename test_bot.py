from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx

import bot


def fake_client(content: str, finish_reason: str = "stop") -> MagicMock:
    client = MagicMock()
    message = SimpleNamespace(content=content)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
    )
    return client


def reply_json(**fields) -> str:
    return bot.LLMReply(**fields).model_dump_json()


def test_symptom_question_is_refused_without_calling_llm():
    client = MagicMock()
    result = bot.answer("I have a fever and chest pain, what should I take?", client=client)
    assert result.source == "guardrail"
    assert result.route_to_staff
    client.chat.completions.create.assert_not_called()


def test_retrieve_finds_hours():
    assert bot.retrieve("When are you open on Saturday?")[0]["id"] == "hours"


def test_no_api_key_uses_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = bot.answer("Where is the clinic located?")
    assert result.source == "fallback"
    assert "Maple Street" in result.answer


def test_valid_llm_reply_is_used():
    content = reply_json(intent="booking", answer="Book online.", route_to_staff=False, sources=["booking"])
    result = bot.answer("How do I book an appointment?", client=fake_client(content))
    assert result.source == "llm"
    assert result.sources == ["booking"]


def test_invalid_json_falls_back():
    result = bot.answer("What are your hours?", client=fake_client('{"intent": "hours"'))
    assert result.source == "fallback"


def test_llm_citing_unknown_fact_falls_back():
    content = reply_json(intent="hours", answer="Open 24/7!", route_to_staff=False, sources=["made_up"])
    result = bot.answer("What are your hours?", client=fake_client(content))
    assert result.source == "fallback"
    assert result.sources == ["hours"]


def test_truncated_response_falls_back():
    content = reply_json(intent="hours", answer="x", route_to_staff=False, sources=["hours"])
    result = bot.answer("What are your hours?", client=fake_client(content, finish_reason="length"))
    assert result.source == "fallback"


def test_api_error_falls_back():
    client = MagicMock()
    client.chat.completions.create.side_effect = groq.APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com")
    )
    result = bot.answer("What are your hours?", client=client)
    assert result.source == "fallback"


def test_unknown_question_routes_to_staff(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = bot.answer("Do you sell gift cards?")
    assert result.route_to_staff
