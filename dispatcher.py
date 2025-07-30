import os, json, re
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, RateLimitError
import backoff

# ------------------------------- კლიენტი  (Moonshot ან OpenAI, როგორც გსურთ)
client = OpenAI(
    base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
    api_key=os.getenv("MOONSHOT_API_KEY"),
)
MODEL = "kimi-k2-0711-preview"

# ----------------------------------------- FastAPI
app = FastAPI()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_to_module",
        "description": "psychology, legal, faq, fallback",
        "parameters": {
            "type": "object",
            "properties": {
                "module":  {"type":"string","enum":["psychology","legal","faq","fallback"]},
                "payload": {"type":"object"},
            },
            "required": ["module","payload"],
        },
    },
}]

SYSTEM_MSG = {"role":"system","content":
    "შენ ხარ მრავალპროფილიანი, პროფესიონალური ასისტენტი. "
    "თერაპია/ტრავმა/პანიკა — გამოიყენე route_to_module(psychology)."
}

PATTERNS = [r"\bთერაპ(ი|ევტ|ია)\b", r"\bსეანს(ი|ები)\b",
            r"\bტრავმ(ა|ული)\b",    r"\bპანიკ(ის|ური)\b"]
def need_psy(s:str)->bool: return any(re.search(p,s,re.I) for p in PATTERNS)

# ------------------------------- /chat
@backoff.on_exception(backoff.expo, RateLimitError, max_time=30)
def kimi(**kw): return client.chat.completions.create(**kw)

@app.post("/chat")
async def chat(r: Request):
    data = await r.json()
    text = data.get("message","").strip()
    if not text: return {"error":"empty"}
    if not client.api_key: return {"error":"API key missing"}

    # ① პირველი მოდელის პასუხი
    first = kimi(model=MODEL,
                 messages=[SYSTEM_MSG,{"role":"user","content":text}],
                 tools=TOOLS, tool_choice="auto")
    a = first.choices[0].message
    tc = a.tool_calls or []

    # ② თუ არც tool არც ჰეურისტიკა − პირდაპირ
    if not tc and not need_psy(text):
        return {"reply": a.content}

    # ③ ჰეურისტიკული ფსიქო‑ტრიგერი
    if not tc:
        tc = [{
            "id": "auto_psy",
            "function": {"name": "route_to_module"},
            "arguments": json.dumps({"module":"psychology","payload":{"text":text}})
        }]

    call = tc[0]
    args = json.loads(call["arguments"]) if isinstance(call,dict) else \
           json.loads(call.function.arguments)
    result = await handle_module(args["module"], args["payload"])

    # ④ ორიგინალი assistant‑ი + შესაბამისი function‑პასუხი
    follow = [
        a.model_dump(),   # content=None, tool_calls=[…]
        {
            "role":"function",
            "name": call["function"]["name"] if isinstance(call,dict) else call.function.name,
            "tool_call_id": call["id"]       if isinstance(call,dict) else call.id,
            "content": json.dumps(result,ensure_ascii=False),
        }
    ]

    final = kimi(model=MODEL, messages=follow, tool_choice="none")
    return {"reply": final.choices[0].message.content}

# ------------------------------- შიდა მოდულები
async def handle_module(mod:str, payload:dict)->dict:
    if mod=="psychology":
        return {"module":"psychology","result":"ok",
                "advice":"ტესტური პასუხი ფსიქო‑მოდულიდან","payload":payload}
    return {"module":mod,"result":"ok","payload":payload}

# ------------------------------- Health + Static UI
@app.get("/health")
async def health(): return {"status":"ok"}
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    # სრულ stacktrace‑ს ლოგში ვტოვებთ, front‑end‑ს მოკლე ტექსტი უბრუნდება
    return JSONResponse(500, content={"error": str(exc)})
