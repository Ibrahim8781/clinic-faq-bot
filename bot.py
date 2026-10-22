"""Clinic FAQ bot: retrieval + Groq structured output, with a guardrail and a fallback."""

import json
import os
import re
from pathlib import Path
from typing import Literal

import groq
from pydantic import BaseModel, ConfigDict, ValidationError

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FACTS = json.loads((Path(__file__).parent / "clinic_facts.json").read_text())["facts"]
STAFF_PHONE = "(555) 010-2000"

Intent = Literal["hours", "location", "booking", "symptom", "other"]


class LLMReply(BaseModel):
    """The shape the LLM must return."""

    model_config = ConfigDict(extra="forbid")  # strict JSON schema needs additionalProperties: false

    intent: Intent
    answer: str
    route_to_staff: bool
    sources: list[str]


class BotAnswer(LLMReply):
    """What the service returns: the reply plus where it came from."""

    source: Literal["llm", "fallback", "guardrail"]


# --- Guardrail: symptom / medical questions never reach the LLM ---

SYMPTOM_PATTERN = re.compile(
    r"\b(pain|ache|hurt|fever|rash|cough|sick|dizz|nause|vomit|bleed|swell|infect|"
    r"symptom|diagnos|headache|chest|breath|dose|dosage|medication|medicine|pill)",
    re.IGNORECASE,
)


def is_symptom(question: str) -> bool:
    return bool(SYMPTOM_PATTERN.search(question))


def guardrail_answer() -> BotAnswer:
    return BotAnswer(
        intent="symptom",
        answer=(
            "I can't give medical advice. I've routed your question to our staff - "
            f"call {STAFF_PHONE}, or 911 for emergencies."
        ),
        route_to_staff=True,
        sources=[],
        source="guardrail",
    )


# --- Retrieval: simple word overlap over the fact chunks ---

STOPWORDS = {"a", "an", "the", "is", "are", "do", "does", "you", "your", "we", "i", "my",
             "to", "of", "on", "in", "at", "for", "and", "or", "can", "it", "what", "how"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower())) - STOPWORDS


def retrieve(question: str, k: int = 2) -> list[dict]:
    q = _words(question)
    scored = []
    for fact in FACTS:
        score = 2 * len(q & set(fact["keywords"])) + len(q & _words(fact["text"]))
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]


# --- Fallback: answer straight from the top fact, no LLM ---

def fallback_answer(facts: list[dict]) -> BotAnswer:
    if not facts:
        return BotAnswer(
            intent="other",
            answer=f"I'm not sure about that - I've routed you to our staff. Call {STAFF_PHONE}.",
            route_to_staff=True,
            sources=[],
            source="fallback",
        )
    top = facts[0]
    return BotAnswer(
        intent=top["intent"],
        answer=top["text"],
        route_to_staff=False,
        sources=[top["id"]],
        source="fallback",
    )


# --- LLM path ---

SYSTEM_PROMPT = f"""You are the front-desk FAQ assistant for a small clinic.
Answer ONLY using the facts provided in the user message. Keep answers to one or two sentences.
Restate the relevant facts; don't guess about details the user didn't give (like dates or times).
List the ids of the facts you used in `sources`.
If the facts don't cover the question, say you don't know, use intent "other" and set route_to_staff to true.
Never give medical advice. If the question is about symptoms, health, or treatment, use intent "symptom",
set route_to_staff to true, and tell the user staff will help at {STAFF_PHONE}."""


def ask_llm(question: str, facts: list[dict], client: groq.Groq) -> BotAnswer:
    context = "\n".join(f"[{f['id']}] {f['text']}" for f in facts) or "(no matching facts)"
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "clinic_reply", "strict": True, "schema": LLMReply.model_json_schema()},
        },
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop" or not choice.message.content:
        raise ValueError(f"unusable LLM response (finish_reason={choice.finish_reason})")

    # Validate the JSON ourselves, then apply business rules the schema can't express.
    reply = LLMReply.model_validate_json(choice.message.content)
    allowed_ids = {f["id"] for f in facts}
    if not set(reply.sources) <= allowed_ids:
        raise ValueError(f"LLM cited facts it wasn't given: {reply.sources}")
    if reply.intent == "symptom":
        reply.route_to_staff = True
    return BotAnswer(**reply.model_dump(), source="llm")


def answer(question: str, client: groq.Groq | None = None) -> BotAnswer:
    if is_symptom(question):
        return guardrail_answer()

    facts = retrieve(question)
    if client is None:
        if not os.getenv("GROQ_API_KEY"):
            return fallback_answer(facts)
        client = groq.Groq()

    try:
        return ask_llm(question, facts, client)
    except (groq.APIError, ValidationError, ValueError) as err:
        print(f"[fallback] {type(err).__name__}: {err}")
        return fallback_answer(facts)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    print("Clinic FAQ bot - ask about hours, location or booking (Ctrl+C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print(answer(q).model_dump_json(indent=2))
