# backend/dispatcher.py
# ===============================================================
#  მთავარი დისპეჩერი – Moonshot-ზე (kimi-k2-0711-preview) დაფუძნებული
#  მრავალმოდულიანი ჩატი.  ფსიქოლოგიური მოთხოვნები ავტომატურად
#  იროუტება სპეც-მოდულზე; დანარჩენი - პირდაპირ მოდელი პასუხობს.
# ===============================================================

import os, json, re
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI, RateLimitError          # type: ignore
import backoff

# ----------------------------------------------------------------- Moonshot Kimi
client = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
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
                "module":  {"type": "string",
                            "enum": ["psychology", "legal", "faq", "fallback"]},
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
        "თერაპია, პანიკა, ტრავმა ან მსგავსი თემის შემთხვევაში აუცილებლად "
        "route_to_module(psychology) უნდა გამოიძახო. "
        "სხვა დროს ინსტრუმენტს გამოიყენებ მხოლოდ საჭიროების შემთხვევაში."
    )
}

# --- ჰეურისტიკული ტრიგერები ---------------------------------------------------
_PATTERNS = [
    r"\bთერაპ(ი|ევტ|ია)\b", r"\bსეანს(ი|ები)\b",
    r"\bტრავმ(ა|ული)\b",    r"\bპანიკ(ის|ური)\b",
]
def need_psy(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _PATTERNS)

# --- 429-ის ავტომატური რეტრაი --------------------------------------------------
@backoff.on_exception(backoff.expo, RateLimitError, max_time=30)
def kimi(**kwargs):
    """Moonshot-ზე გადამისამართებული chat.completions.create ისრობს 429-ზე re-try-ს."""
    return client.chat.completions.create(**kwargs)

# --------------------------------------------------------------------- /chat ---
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    txt = body.get("message", "").strip()
    if not txt:
        return {"error": "empty message"}
    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # ① პირველი რაუნდი — მოდელი თვითონ გადაწყვეტს გამოიძახოს თუ არა tool
    first = kimi(
        model=MODEL,
        messages=[SYSTEM_MSG, {"role": "user", "content": txt}],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.8, top_p=0.95, presence_penalty=0.5,
    )

    assistant = first.choices[0].message
    tcalls    = assistant.tool_calls or []

    # ② არც tool, არც ჰეურისტიკა: პასუხის პირდაპირ გადაცემა
    if not tcalls and not need_psy(txt):
        return {"reply": assistant.content}

    # ③ ჰეურისტიკით გამოწვეული ფსიქო-მოდულის გამოთხოვა
    if not tcalls:
        tcalls = [{
            "id": "auto_psychology",
            "function": {"name": "route_to_module"},
            "arguments": json.dumps({"module": "psychology",
                                     "payload": {"text": txt}})
        }]

    call = tcalls[0]
    # arguments → dict
    args = json.loads(call["arguments"]) if isinstance(call, dict) \
        else json.loads(call.function.arguments)

    # ④ შიდა მოდულის დამუშავება
    result = await handle_module(args["module"], args["payload"])

    # ⑤ ვაგზავნით assistant-ისა და function-ის წყვილს ისევ Kimi-ში
    follow_msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tcalls,
        },
        {
            "role": "function",
            "name":  call["function"]["name"] if isinstance(call, dict)
                     else call.function.name,
            "tool_call_id": call["id"] if isinstance(call, dict) else call.id,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]

    final = kimi(
        model=MODEL,
        messages=follow_msgs,
        tool_choice="none",
        temperature=0.8, top_p=0.95, presence_penalty=0.5,
    )
    return {"reply": final.choices[0].message.content}

# ----------------------------------------------------------- შიდა მოდულები ---
async def handle_module(mod: str, payload: dict) -> dict:
    """ამ ეტაპზე – სტუბი; აქ შეიტანეთ რეალური ფსიქო-ასისტენტის ლოგიკა."""
    if mod == "psychology":
        return {
            "module": "psychology",
            "result": "ok",
            "advice": "ეს არის ტესტური პასუხი ფსიქო-მოდულიდან.",
            "payload": payload,
        }
    return {"module": mod, "result": "ok", "payload": payload}

# ------------------------------------------------------------- Health + Static
@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# -------------------------------------------------------- გლობალური catcher ---
@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    # ლოგში სრულ stacktrace-ს ევროლიმ აისახება; front-end-ს მხოლოდ ტექსტი.
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}"}
    )
