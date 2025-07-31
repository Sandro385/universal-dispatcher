# backend/dispatcher.py
# ---------------------------------------------------------------------------
import os
import json
import re
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI, RateLimitError
import backoff

# -------------------------------------------------------------------- logging
logging.basicConfig(level=logging.DEBUG)

# -------------------------------------------------------------------- Moonshot
client = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
)
MODEL = "kimi-k2-0711-preview"

# ---------------------------------------------------------------------- FastAPI
app = FastAPI()

TOOLS: List[Dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "route_to_module",
        "description": "psychology, legal, faq, fallback",
        "parameters": {
            "type": "object",
            "properties": {
                "module":  {
                    "type": "string",
                    "enum": ["psychology", "legal", "faq", "fallback"]
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

# ჰეურისტიკა
_PATTERNS = [
    r"\bთერაპ(ი|ევტ|ია)\b", r"\bსეანს(ი|ები)\b",
    r"\bტრავმ(ა|ული)\b",    r"\bპანიკ(ის|ური)\b",
]
def need_psychology(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _PATTERNS)

# --------------------------------------------------------------------------- util
@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def kimi(**kw):
    """Moonshot call with automatic backoff on 429."""
    return client.chat.completions.create(**kw)

# --------------------------------------------------------------------------- /chat
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_text: str = data.get("message", "").strip()

    if not user_text:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if not client.api_key:
        return JSONResponse({"error": "MOONSHOT_API_KEY is not configured"}, status_code=500)

    base_msgs = [SYSTEM_MSG, {"role": "user", "content": user_text}]

    # ① Kimi პირველი პასუხი
    first = kimi(model=MODEL,
                 messages=base_msgs,
                 tools=TOOLS,
                 tool_choice="auto")

    assistant = first.choices[0].message
    tool_calls = assistant.tool_calls or []

    # ② თუ არც tool და არც ჰეურისტიკა
    if not tool_calls and not need_psychology(user_text):
        return {"reply": assistant.content}

    # ③ ჰეურისტიკული განშტოება
    if not tool_calls:
        tool_calls = [{
            "id": "auto_psychology",
            "type": "function",
            "function": {
                "name": "route_to_module",
                "arguments": json.dumps(
                    {"module": "psychology", "payload": {"text": user_text}},
                    ensure_ascii=False
                )
            }
        }]

    # ④ arguments გაშლა
    call = tool_calls[0]
    if isinstance(call, dict):
        args = json.loads(call["function"]["arguments"])
        call_id   = call["id"]
        func_name = call["function"]["name"]
    else:
        args = json.loads(call.function.arguments)
        call_id   = call.id
        func_name = call.function.name

    # ⑤ შიდა მოდულის შესრულება
    module_result = await handle_module(args["module"], args["payload"])

    # ⑥ assistant + tool messages (OpenAI-ს სწორი ფორმატით)
    assistant_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            }
        ]
    }
    tool_response = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(module_result, ensure_ascii=False),
    }

    follow_msgs = base_msgs + [assistant_call, tool_response]

    # ⑦ საბოლოო ტექსტის მიღება
    final = kimi(model=MODEL,
                 messages=follow_msgs,
                 tool_choice="none")

    return {"reply": final.choices[0].message.content}

# -------------------------------------------------------------------- Modules
async def handle_module(mod: str, payload: dict) -> dict:
    """საჭიროებისამებრ ჩაანაცვლე რეალური ლოგიკით."""
    if mod == "psychology":
        return {
            "module": "psychology",
            "result": "ok",
            "advice": "ტესტური პასუხი ფსიქო-მოდულიდან",
            "payload": payload,
        }
    return {"module": mod, "result": "ok", "payload": payload}

# ----------------------------------------------------------------- Health + UI
@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
