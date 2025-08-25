"""
A minimal FastAPI dispatcher service for testing psychological routing.

This app is adapted from the user's code snippet.  It tracks per-session
state and forwards messages either to a general assistant (Moonshot) or
to a psychological module.  For testing purposes, calls to OpenAI's API
have been stubbed out so the service can run in this environment without
external dependencies.
"""

import logging
import os
import re
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GENERAL_SYSTEM_PROMPT = (
    "შენ ხარ მეგობრული და მრავალმხრივი AI ასისტენტი (Moonshot). "
    "პასუხები უნდა იყოს მოკლე, პროფესიონალური და მომხმარებელზე მორგებული. "
    "გამოიყენე ქართულად საუბარი და საჭიროების შემთხვევაში ტექსტი გადმოაგზავნე სათანადო ფორმით. "
)

PSYCH_SYSTEM_PROMPT = (
    "შენ ხარ გამოცდილი ფსიქოლოგი. "
    "მომხმარებელს დაეხმარე ემოციურად, უსაფრთხოდ და კონფიდენციალურად. "
    "როცა იგრძნობ, რომ მომხმარებელს აღარ სჭირდება ფსიქოლოგიური დახმარება და საუბარი გადავიდა ზოგად საკითხებზე, "
    "პასუხის ბოლოს დაამატე სპეციალური მარკერი '[handoff]' (ბრჭყალების გარეშე), რათა დისპეჩერმა იცოდეს, რომ უნდა დაბრუნდეს ზოგად რეჟიმში. "
)

# Keywords used for quick detection of psychological topics
PSYCH_KEYWORDS = re.compile(
    r"ფსიქ|თერაპ|პსიქ|დეპრეს|depress|anxiety|therapy|psych", re.IGNORECASE
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state: holds current module and history for each session
session_state: Dict[str, Dict[str, Any]] = {}

async def classify_intent_via_llm(text: str) -> float:
    """Stubbed classification function: returns 0.0 for all inputs.

    In production this would call OpenAI's API and return a value between
    0 and 1 indicating the probability that the text requires
    psychological support.  Here we always return 0 so that only the
    keyword pattern triggers routing.
    """
    return 0.0

async def detect_module(text: str) -> str:
    """Determine which module should handle the given text.

    1. Use a quick keyword match for psychological terms.
    2. Otherwise, use an LLM-based classifier (stubbed here).
    Probability ≥ 0.3 -> "psychology", else -> "general".
    """
    if PSYCH_KEYWORDS.search(text or ""):
        return "psychology"
    probability = await classify_intent_via_llm(text or "")
    return "psychology" if probability >= 0.3 else "general"

async def call_openai_chat(messages: List[Dict[str, str]]) -> str:
    """Stubbed call to OpenAI's chat API.

    For this test environment we simply return a canned response.
    In a real deployment, this function would call OpenAI's API and
    return the assistant's response text.
    """
    # Join the last user message with the system prompt to build a dummy reply
    user_content = next((m["content"] for m in messages[::-1] if m["role"] == "user"), "")
    return f"Stubbed reply to: {user_content}"

async def handle_general(session: Dict[str, Any], user_text: str) -> str:
    """Handle a message in the general (Moonshot) module.

    Maintains history and returns a response from the stubbed model.
    """
    history = session.setdefault("history", [])
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # limit history length
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

async def handle_psychology(session: Dict[str, Any], user_text: str) -> str:
    """Handle a message in the psychology module.

    This stub simply returns a reply and never triggers a handoff.  In a
    real deployment, this would call a remote psychology service or use
    an LLM with a psychology-specific system prompt.  If the response
    ends with [handoff], session["current_module"] would be set to
    "general".
    """
    history = session.setdefault("history", [])
    messages = [
        {"role": "system", "content": PSYCH_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # Check for handoff marker in stubbed response
    if "[handoff]" in response:
        response = response.replace("[handoff]", "").strip()
        session["current_module"] = "general"
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

@app.post("/chat")
async def chat(request: Request):
    """Main chat endpoint.

    Body should contain: {"text": "...", "module": "...", "session_id": "..."}.
    If module is omitted, the dispatcher chooses based on session state
    and intent detection.
    """
    body = await request.json()
    user_text = body.get("text", "")
    forced_module = body.get("module")
    session_id = body.get("session_id", "default")

    if not user_text:
        return JSONResponse(status_code=400, content={"error": "text field is required"})

    # Retrieve or initialize session state
    session = session_state.setdefault(session_id, {"current_module": "general", "history": []})
    session["id"] = session_id

    # Determine module: forced, existing, or detected
    if forced_module:
        module = forced_module
    else:
        module = session.get("current_module", "general")
        # If currently in general mode, detect whether to switch to psychology
        if module == "general":
            module = await detect_module(user_text)

    # Save the module pre-dispatch
    session["current_module"] = module

    # Dispatch to appropriate handler
    if module == "psychology":
        assistant_text = await handle_psychology(session, user_text)
        # Update module from session in case the psychology handler triggered a handoff
        module = session.get("current_module", module)
    else:
        assistant_text = await handle_general(session, user_text)

    return {"module": module, "text": assistant_text}

@app.get("/health")
async def health():
    return {"status": "ok"}
