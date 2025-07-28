import os
import json

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# Attempt to import the real OpenAI client; fall back to stub if missing
try:
    from openai import OpenAI        # type: ignore
except ImportError:
    from openai_stub import OpenAI   # type: ignore

# --- ✅ REGION‑სატენიერო პარამეტრი ------------------------------
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
# ----------------------------------------------------------------

client = OpenAI(
    base_url=MOONSHOT_BASE_URL,           # <‑‑ default .ai, ან ENV‑ით გადაწერეთ
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

app = FastAPI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_to_module",
            "description": "არჩევს შესაბამის მოდულს მომხმარებლის მოთხოვნისთვის",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string",
                               "enum": ["psychology", "legal", "faq", "fallback"]},
                    "payload": {"type": "object"},
                },
                "required": ["module", "payload"],
            },
        },
    }
]

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_msg = body.get("message", "")

    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": user_msg}],
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as exc:
        return {"error": f"სერვერის შეცდომა: {type(exc).__name__}"}

    tcalls = response.choices[0].message.tool_calls
    if tcalls:
        data = json.loads(tcalls[0].function.arguments)
        return await handle_module(data["module"], data["payload"])

    return {"reply": response.choices[0].message.content}

async def handle_module(mod: str, payload: dict):
    return {"module": mod,
            "result": f"✅ {mod}-მოდული აღძრულია",
            "payload": payload}

@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
