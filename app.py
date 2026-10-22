from fastapi import FastAPI
from pydantic import BaseModel

import bot

app = FastAPI(title="Clinic FAQ Bot")


class Question(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=bot.BotAnswer)
def ask(body: Question) -> bot.BotAnswer:
    return bot.answer(body.question)
