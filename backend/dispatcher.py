import os
import json
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# Attempt to import the real OpenAI client.  If the package is not
# available (e.g. due to network restrictions during installation), fall
# back to a local stub implementation that simply echoes messages.
try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - executed only when openai is missing
    from openai_stub import OpenAI  # type: ignore

# Initialize OpenAI client using environment variable for the API key.  This key
# should be provided at runtime via the environment (e.g. in Render).  Without
# a valid key the application will not be able to call the Moonshot model.
client = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY")
)

# Create FastAPI application instance
app = FastAPI()

# NOTE: we mount the static frontend at the end of the module after all other
# routes are defined.  Mounting at the root path here would preempt any
# subsequently defined routes (e.g. ``/chat`` or ``/health``), causing them
# to return a 404.  See the bottom of this file for the mount call.

# Define tools for function‑calling behaviour.  These tools describe how the
# model should respond when the user input requires routing to a specific
# module.  The ``route_to_module`` tool returns a module name and arbitrary
# payload which is then handled by ``handle_module``.
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
                        "enum": ["psychology", "legal", "faq", "fallback"]
                    },
                    "payload": {"type": "object"}
                },
                "required": ["module", "payload"]
            }
        }
    }
]


@app.post("/chat")
async def chat(request: Request):
    """Accept a JSON payload with a ``message`` field and return a reply.

    The user's message is forwarded to the Moonshot API.  If the model chooses
    to call a tool, the call is dispatched to ``handle_module``; otherwise the
    raw text response is returned.
    """
    body = await request.json()
    user_msg = body.get("message", "")

    # Make sure the API key is present before attempting to call the model.  If
    # it's missing we return an error to the client rather than raising an
    # exception.  This helps during development when the environment variable
    # might not be set.
    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # Use OpenAI's chat API with the function calling configuration.  We pass
    # the user message as the first message and allow the model to decide
    # whether a tool should be invoked.
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOLS,
        tool_choice="auto",
    )

    # If the model invoked a tool, call our internal handler; otherwise return
    # the content of the model's message.
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        call = tool_calls[0]
        data = json.loads(call.function.arguments)
        return await handle_module(data["module"], data["payload"])

    return {"reply": response.choices[0].message.content}


async def handle_module(mod: str, payload: dict):
    """Handle a request routed to a specific module.

    This is a stub implementation; in a real application you could implement
    custom logic for each module (e.g. call a separate API, perform some
    database operations, etc.).  For now it returns a simple JSON object
    indicating which module was chosen and echoes the provided payload.
    """
    return {
        "module": mod,
        "result": f"✅ {mod}-მოდული აღძრულია",
        "payload": payload,
    }


@app.get("/health")
async def health():
    """Health check endpoint used by Render to verify the service is up."""
    return {"status": "ok"}


# At the end of this module we mount the static frontend at ``/``.  Mounting
# after the route declarations ensures that our API endpoints (``/chat`` and
# ``/health``) take precedence over the static file handler.  Without this
# ordering FastAPI would dispatch every request (including API endpoints) to
# the static handler, resulting in 404 responses for the API.
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


if __name__ == "__main__":
    # When running locally (python dispatcher.py), use uvicorn to start the
    # application.  This block isn't executed when uvicorn is started via
    # ``uvicorn backend.dispatcher:app`` in the Dockerfile CMD.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)