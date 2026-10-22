from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest

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
    result = bot.answer("I have a fever and a cough, what should I take?", client=client)
    assert result.source == "guardrail"
    assert result.intent == "symptom"
    assert result.route_to_staff
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("question", [
    "I think I'm having a heart attack",
    "My dad has chest pain",
    "I can’t breathe",  # curly apostrophe, as phones type it
    "She passed out and is unconscious",
    "I think I'm in labour",
])
def test_emergency_tells_patient_to_call_emergency_services(question):
    client = MagicMock()
    result = bot.answer(question, client=client)
    assert result.intent == "emergency"
    assert result.source == "guardrail"
    assert bot.EMERGENCY_NUMBER in result.answer
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("question", [
    "Do you do chest X-rays?",
    "Do you have a heart specialist?",
    "Do you offer labour and delivery care?",
])
def test_service_questions_are_not_blocked(monkeypatch, question):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = bot.answer(question)
    assert result.source != "guardrail"


def test_llm_flagged_symptom_uses_fixed_safety_wording():
    # The regex misses this one, so the LLM is the second line of defence.
    content = reply_json(intent="symptom", answer="Please contact our staff.", route_to_staff=True, sources=[])
    result = bot.answer("My chest feels tight and my arm is numb", client=fake_client(content))
    assert result.source == "llm"
    assert result.route_to_staff
    assert bot.EMERGENCY_NUMBER in result.answer


def test_emergency_number_comes_from_config(monkeypatch):
    monkeypatch.setattr(bot, "EMERGENCY_NUMBER", "112")
    assert "112" in bot.answer("I think I'm having a heart attack").answer
    assert "112" in bot.answer("I have a rash").answer
    assert "999" in next(f["text"] for f in bot.FACTS if f["id"] == "contact")


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
