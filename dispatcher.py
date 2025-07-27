import os
import json

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# Attempt to import the real OpenAI client; fall back to stub if missing
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    from openai_stub import OpenAI  # type: ignore

# Initialize client using environment variable for API key
client = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

# Create FastAPI application
app = FastAPI()

# Define tools for function-calling routing
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
    """Handle chat requests by forwarding to the Moonshot API."""
    body = await request.json()
    user_msg = body.get("message", "")

    # Ensure API key is present
    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # Attempt to call the Moonshot API. Wrap in try/except so that errors are returned as JSON.
    try:
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": user_msg}],
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as exc:
        # Return only the exception type to avoid leaking details
        return {"error": f"სერვერის შეცდომა: {type(exc).__name__}"}

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        call = tool_calls[0]
        data = json.loads(call.function.arguments)
        return await handle_module(data["module"], data["payload"])

    return {"reply": response.choices[0].message.content}

async def handle_module(mod: str, payload: dict):
    """Stub handler for routed modules. Returns module and payload."""
    return {
        "module": mod,
        "result": f"✅ {mod}-მოდული აღძრულია",
        "payload": payload,
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

# Mount static frontend after API routes to avoid 404s on /chat and /health
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    # Use the PORT environment variable if provided (Render sets this), otherwise default to 8000
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
