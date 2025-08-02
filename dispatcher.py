# backend/dispatcher.py
"""Moonshot‑ზე დამყარებული მრავალმოდულიანი დისპეჩერი
ხელახლა დაწერილი — ზარმაცი (lazy) კლიენტების ინიციალიზაციით და
არაბლოკური async/await გამოძახებით.
"""
import os
import json
import re
import logging
import asyncio
import functools
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI, AsyncOpenAI, RateLimitError
import backoff

logging.basicConfig(level=logging.DEBUG)

# ------------------------------------------------------------------
# Lazy factories — კლიენტები იძახება მხოლოდ საჭიროებისას
# ------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_moonshot_client() -> OpenAI:
    key = os.getenv("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY env var is missing")
    return OpenAI(
        base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        api_key=key,
    )

MOONSHOT_MODEL = "kimi-k2-0711-preview"


@functools.lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is missing")
    return AsyncOpenAI(api_key=key)  # default base_url

OPENAI_MODEL = "gpt-4o-mini"

# ------------------------------------------------------------------
# FastAPI აპი
# ------------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------------
# Tool-ები Moonshot-ისთვის
# ------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "route_to_module",
        "description": "psychology, legal, faq, fallback",
        "parameters": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "enum": ["psychology", "legal", "faq", "fallback"],
                },
                "payload": {"type": "object"},
            },
            "required": ["module", "payload"],
        },
    },
}]

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "შენ ხარ მრავალპროფილიანი, პროფესიონალური ასისტენტი. "
        "თუ საუბარი ეხება თერაპიას, პანიკურ შეტევას, ძლიერ ტრავმულ გამოცდილებას "
        "ან სხვა ინტენსიურ ფსიქოლოგიურ თემას, გამოიძახე "
        "route_to_module მოდული = psychology. "
        "დანარჩენ შემთხვევაში ინსტრუმენტი გამოიყენე მხოლოდ მაშინ, "
        "თუ სპეციალისტის ჩართვა ნამდვილად საჭიროა."
    ),
}

_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bთერაპ(?:ი|ევტ|ია)\b",
    r"\bსეანს(?:ი|ები)\b",
    r"\bტრავმ(?:ა|ული)\b",
    r"\bპანიკ(?:ის|ური)\b",
]]

def need_psychology(text: str) -> bool:
    """რეგექსებით ფსიქოლოგიური თემის მარტივი სიგნალი."""
    return any(p.search(text) for p in _PATTERNS)

# ------------------------------------------------------------------
# Utility wrappers
# ------------------------------------------------------------------
@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def _kimi_sync(**kw):
    """Moonshot synchronous call with backoff."""
    client = get_moonshot_client()
    return client.chat.completions.create(**kw)

async def kimi(**kw):
    """Non‑blocking Moonshot call suitable for async endpoints."""
    return await asyncio.to_thread(_kimi_sync, **kw)

# ------------------------------------------------------------------
# /chat – Moonshot-ის დისპეჩერი
# ------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_text: str = data.get("message", "").strip()

    if not user_text:
        return JSONResponse({"error": "empty message"}, status_code=400)

    base_msgs = [SYSTEM_MSG, {"role": "user", "content": user_text}]

    # ① Moonshot-ის პირველი პასუხი
    first = await kimi(
        model=MOONSHOT_MODEL,
        messages=base_msgs,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant = first.choices[0].message
    tool_calls = getattr(assistant, "tool_calls", None) or []

    # თუ სპეციალისტი არ არის საჭირო
    if not tool_calls and not need_psychology(user_text):
        return {"reply": assistant.content}

    # fallback: regex-მა მოითხოვა ფსიქოლოგი
    if not tool_calls:
        tool_calls = [{
            "id": "auto_psychology",
            "type": "function",
            "function": {
                "name": "route_to_module",
                "arguments": json.dumps({
                    "module": "psychology",
                    "payload": {"text": user_text},
                }, ensure_ascii=False),
            },
        }]

    call = tool_calls[0]
    args = json.loads(call["function"]["arguments"]) if isinstance(call, dict) else json.loads(call.function.arguments)

    # ② შიდა მოდულის გამოძახება
    module_result = await handle_module(args["module"], args["payload"])

    # ③ საბოლოო პასუხი მომხმარებლისთვის
    return {"reply": module_result.get("text", "მოდულმა პასუხი არ დააბრუნა")}

# ------------------------------------------------------------------
# Modules – OpenAI-ს ფსიქოლოგიური მოდული
# ------------------------------------------------------------------
async def handle_module(mod: str, payload: dict) -> dict:
    if mod == "psychology":
        client = get_openai_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "შენ ხარ გამოცდილი ფსიქოლოგი. "
                    "მომხმარებელს დაეხმარე ემოციურად, უსაფრთხოდ და კონფიდენციალურად."
                ),
            },
            {"role": "user", "content": payload["text"]},
        ]

        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
            )
            return {
                "module": "psychology",
                "text": resp.choices[0].message.content,
            }
        except Exception as e:
            logging.exception("OpenAI psychology module error")
            return {"error": str(e)}

    # დანარჩენი მოდულები — სტუბები
    return {"module": mod, "text": f"სტუბ-პასუხი {mod}-დან"}

# ------------------------------------------------------------------
# Health + UI
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
