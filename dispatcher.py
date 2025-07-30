import os, json, re, time
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# -------------------------------------------------------------------- Moonshot
try:
    from openai import OpenAI                 # type: ignore
except ImportError:
    from openai_stub import OpenAI            # type: ignore

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
        "description": "არჩევს შესაბამის მოდულს (psychology, legal, faq …)",
        "parameters": {
            "type": "object",
            "properties": {
                "module":  {"type": "string", "enum": ["psychology","legal","faq","fallback"]},
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
        "თერაპია/სეანსი/პანიკა/ტრავმა — აუცილებლად route_to_module(psychology). "
        "ხოლო ინსტრუმენტი გამოიყენე მხოლოდ რეალურად აუცილებელ შემთხვევაში."
    ),
}

PSYCHO_PATTERNS = [
    r"\bთერაპ(ი|ევტ|ია)\b", r"\bსეანს(ი|ები)\b",
    r"\bტრავმ(ა|ული)\b",     r"\bდეპრეს(ია|იული)\b",
    r"\bპანიკ(ის|ური)\b",    r"\bდიაგნოზ(ი|ის)\b",
]
def detect_need(txt: str) -> bool:
    return any(re.search(pat, txt, re.IGNORECASE) for pat in PSYCHO_PATTERNS)

# --------------------------------------------------------------------------- API
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_msg: str = body.get("message", "").strip()
    if not user_msg:
        return {"error": "empty message"}
    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    messages: List[Dict[str, Any]] = [SYSTEM_MSG, {"role": "user", "content": user_msg}]

    # ① პირველი გამოძახება
    first = client.chat.completions.create(
        model=MODEL, messages=messages,
        tools=TOOLS, tool_choice="auto",
        temperature=0.8, top_p=0.95, presence_penalty=0.5,
    )
    assistant = first.choices[0].message
    tcalls    = assistant.tool_calls or []

    # --- ინსტრუმენტი არ არსებობდა → პირდაპირ პასუხი --------------------------
    if not tcalls and not detect_need(user_msg):
        return {"reply": assistant.content}

    # --- ინსტრუმენტი არსებობდა, ან ჰეურისტიკამ მოითხოვა ----------------------
    if not tcalls:                                              # ჰეურისტიკის გზა
        tcalls = [{
            "id": "auto_psychology",
            "function": {"name": "route_to_module"},
            "arguments": json.dumps({"module": "psychology", "payload": {"text": user_msg}})
        }]

    call = tcalls[0]
    args = json.loads(call["arguments"]) if isinstance(call, dict) else json.loads(call.function.arguments)
    args["payload"].setdefault("text", user_msg)

    module_result = await handle_module(args["module"], args["payload"])

    # --- tool‑ს პასუხი ვუგზავნით Kimi‑ს ----------------------------
    follow_msgs = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": None, "tool_calls": tcalls},
        {
            "role": "function",
            "name": "route_to_module",
            "tool_call_id": call["id"] if isinstance(call, dict) else call.id,   # ★ დამატებულია
            "content": json.dumps(module_result, ensure_ascii=False),
        },
    ]

    final = client.chat.completions.create(
        model=MODEL,
        messages=follow_msgs,
        tool_choice="none",
        temperature=0.8, top_p=0.95, presence_penalty=0.5,
    )
    return {"reply": final.choices[0].message.content}

# -------------------------------------------------------------------- Modules
async def handle_module(mod: str, payload: dict) -> dict:
    # თქვენ აქ ჩასვამთ რეალურ ფსიქო‑ლოგიკას; ჯერჯერობით სტუბი
    return {
        "module": mod,
        "result": "ok",
        "advice": (
            "ეს არის ტესტური პასუხი ფსიქო‑მოდულიდან "
            "(აქ ჩამოაგდებთ OpenAI‑assistant‑ის რეალურ შედეგს)."
        ),
        "payload": payload,
    }

# ----------------------------------------------------------------- Health/UI
@app.get("/health")
async def health(): return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.exception_handler(Exception)
async def err(_, exc: Exception):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
