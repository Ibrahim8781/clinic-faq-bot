# Clinic FAQ Bot

A small demo bot for a fictional clinic ("Maple Street Family Clinic"). It answers questions about **hours, location and booking**. It **refuses symptom or medical questions** and routes them to staff.

## What it demonstrates

| Concept | Where |
|---|---|
| **RAG**: retrieve facts, answer only from them | `retrieve()` in `bot.py` picks facts from `clinic_facts.json` by word overlap and passes them to the LLM as context |
| **LLM-backed service** | `ask_llm()` calls an LLM on Groq; `app.py` serves it as `POST /ask` |
| **Structured-output validation** | The LLM must return JSON matching the `LLMReply` Pydantic schema (Groq strict `json_schema` mode). The code validates that JSON with Pydantic and checks business rules: cited `sources` must be among the retrieved facts, and `symptom` always sets `route_to_staff` |
| **Fallback** | No API key, an API error, a truncated reply or a failed validation each return the best matching fact verbatim (`source: "fallback"`) |
| **Guardrail** | `is_symptom()` blocks medical questions *before* any LLM call (`source: "guardrail"`) |

Request flow: `question -> guardrail -> retrieve -> LLM (structured) -> validate -> answer`. Any failure on the LLM step goes to the fallback.

## Run

```bash
pip install -r requirements.txt
pytest -q                      # runs offline; the LLM is mocked

python bot.py                  # interactive CLI
uvicorn app:app --reload       # HTTP service on :8000
```

Set `GROQ_API_KEY` to use the LLM. Without it the bot still works in fallback mode. To change the model, set `GROQ_MODEL` (default `openai/gpt-oss-20b`). The model must support Groq's strict JSON-schema mode.

```powershell
$env:GROQ_API_KEY = "gsk_..."   # current PowerShell session only; never commit the key
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What time do you open on Saturday?"}'
```

Example response:
```json
{"intent": "hours", "answer": "On Saturday we're open 9:00 AM - 1:00 PM.",
 "route_to_staff": false, "sources": ["hours"], "source": "llm"}
```

## Kept simple on purpose
Retrieval is keyword overlap, not embeddings. With only six facts, a vector DB would add nothing. The upgrade path is to swap `retrieve()` for embedding search. There is no chat memory and no UI.
