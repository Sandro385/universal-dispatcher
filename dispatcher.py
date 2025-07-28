import os
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from openai import OpenAI          # type: ignore
except ImportError:
    from openai_stub import OpenAI     # type: ignore

# რეგიონზე დამოკიდებული ბაზა
MOONSHOT_BASE_URL = os.getenv(
    "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"
)

client = OpenAI(
    base_url=MOONSHOT_BASE_URL,
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
                    "module": {
                        "type": "string",
                        "enum": ["psychology", "legal", "faq", "fallback"],
                    },
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
        return {"error": f"{type(exc).__name__}"}

    try:
        tcalls = response.choices[0].message.tool_calls or []
        if tcalls:
            data = json.loads(tcalls[0].function.arguments)
            return await handle_module(data["module"], data["payload"])
        else:
            return {"reply": response.choices[0].message.content}
    except json.JSONDecodeError:
        return {"error": "Bad function arguments (not JSON)"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}"}

async def handle_module(mod: str, payload: dict):
    return {
        "module": mod,
        "result": f"✅ {mod}-მოდული აღძრულია",
        "payload": payload,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

# Static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# Global exception handler
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}"}
    )
