"""
Dispatcher service responsible for routing user messages to different
specialised assistants.  In addition to the original functionality of
forwarding to a psychological assistant when explicitly requested via
the ``module`` field, this version keeps track of per‑session
conversation state and can automatically switch between a general
assistant (e.g. Moonshot) and a psychological assistant based on the
user's intent.  When the user engages in therapeutic topics the
conversation is delegated to the psychology module until that module
signals that it is ready to hand control back to the general assistant.

Session state is tracked in memory using a simple dictionary keyed by
``session_id`` values supplied in the request body.  Each entry
contains the current active module and the running chat history for
that session.  A very simple keyword‑based classifier is used to
determine when a message should be routed to the psychology module if
no explicit module is provided.  For production deployments you may
wish to replace this with a more sophisticated intent classifier.
"""

import logging
import os
import re
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import openai
from openai import AsyncOpenAI
import httpx

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
# OpenAI API key and model.  The default model is set to a lightweight
# variant for cost considerations but can be overridden via
# environment variables.  Two system prompts are defined below: one for
# general conversation (Moonshot) and one for psychological support.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env variable is required")

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

# Compile simple keyword pattern to detect psychological topics.  This
# is intentionally simplistic – it checks for common Georgian and
# English words related to psychology and therapy.  In real systems you
# would likely use an LLM or trained classifier for better accuracy.
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


def get_openai_client() -> AsyncOpenAI:
    """Instantiate a reusable AsyncOpenAI client."""
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------------
# Session management
# -------------------------------------------------------------------------
# A simple in‑memory store for per‑session state.  Each session holds
# the current active module ("general" or "psychology") and the chat
# history as a list of message dictionaries (role/content).  This store
# will be cleared when the process restarts; if persistence is needed,
# integrate a database or cache here.
session_state: Dict[str, Dict[str, Any]] = {}

# Base URL for the remote psychology chat service.  If this environment
# variable is set, messages targeting the psychology module will be
# forwarded to this service instead of using the built‑in OpenAI
# invocation.  The service is expected to expose a POST `/chat`
# endpoint accepting a JSON payload with at least a `text` field and
# returning a JSON object with a `text` field.  Optionally include
# `session_id` in the payload to allow the downstream service to
# maintain conversation state.
PSYCH_CHAT_URL = os.getenv("PSYCH_CHAT_URL")


def detect_module(text: str) -> str:
    """Determine which module should handle the given text.

    Returns "psychology" if the text contains keywords associated with
    therapy or psychological topics, otherwise returns "general".
    """
    if PSYCH_KEYWORDS.search(text or ""):
        return "psychology"
    return "general"


async def call_openai_chat(messages: List[Dict[str, str]]) -> str:
    """Call the OpenAI chat completion API with the given list of messages.

    This helper normalises the response to a single string, handling
    both string and block formats returned by the SDK.
    """
    client = get_openai_client()
    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )
        msg = resp.choices[0].message
        content = ""
        if isinstance(msg.content, str):
            content = msg.content
        else:
            # When content is returned as a list of blocks
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "")
                else:
                    content += getattr(block, "text", "")
        return content
    except Exception as e:
        logging.exception("OpenAI API call failed")
        return f"ხარვეზი მოხდა: {e}"


async def handle_general(session: Dict[str, Any], user_text: str) -> str:
    """Handle a message with the general assistant (Moonshot).

    Appends the user message to the session history, calls the OpenAI
    API with the general system prompt, updates the history with the
    assistant's response and returns the assistant's text.
    """
    history: List[Dict[str, str]] = session.setdefault("history", [])
    # Build messages: system prompt + existing history + new user message
    messages: List[Dict[str, str]] = ([{"role": "system", "content": GENERAL_SYSTEM_PROMPT}] +
                                     history +
                                     [{"role": "user", "content": user_text}])
    response = await call_openai_chat(messages)
    # Append to history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # Truncate history to prevent unbounded growth (keep last 20 exchanges)
    if len(history) > 40:
        session["history"] = history[-40:]
    return response


async def handle_psychology(session: Dict[str, Any], user_text: str) -> str:
    """Handle a message with the psychology assistant.

    Similar to ``handle_general`` but uses the psychological system
    prompt and looks for a handoff marker in the assistant's response.
    If the marker is present the current module for the session is reset
    to ``general`` after delivering the response.
    """
    # If a remote psychology chat service URL is configured, forward
    # requests to it instead of using the local OpenAI prompt.  This
    # allows integration with a dedicated psych‑chat microservice
    # deployed on Render.  The service should accept JSON and return
    # JSON with at least a "text" field.
    if PSYCH_CHAT_URL:
        try:
            async with httpx.AsyncClient() as client:
                payload = {"text": user_text}
                # include session ID if present to allow downstream state
                sid = session.get("id")
                if sid:
                    payload["session_id"] = sid
                resp = await client.post(PSYCH_CHAT_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # Expect "text" and optional "handoff" flag
                response_text = data.get("text", "")
                handoff_flag = data.get("handoff")
                # If downstream service signals handoff, return to general
                if handoff_flag or "[handoff]" in response_text:
                    session["current_module"] = "general"
                    response_text = response_text.replace("[handoff]", "").strip()
                return response_text
        except Exception:
            logging.exception("Error calling remote psychology chat service; falling back to local model")
            # fall through to local OpenAI logic

    # Default local behaviour: use OpenAI with psychology prompt
    history: List[Dict[str, str]] = session.setdefault("history", [])
    messages: List[Dict[str, str]] = ([{"role": "system", "content": PSYCH_SYSTEM_PROMPT}] +
                                     history +
                                     [{"role": "user", "content": user_text}])
    response = await call_openai_chat(messages)
    # Append to history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # Check for handoff marker in the assistant's response
    if "[handoff]" in response:
        cleaned = response.replace("[handoff]", "").strip()
        session["current_module"] = "general"
        response = cleaned
    # Truncate history
    if len(history) > 40:
        session["history"] = history[-40:]
    return response


# ----------------------------------------------------------
# Main chat endpoint – listens on /chat
# ----------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint to interact with the dispatcher.

    Expected JSON body:
    {
      "text": "გამარჯობა, დახმარება მჭირდება...",
      "module": "psychology",          # optional, forces a module
      "session_id": "user-12345"       # optional, groups conversation state
    }

    If ``module`` is omitted the dispatcher will determine the appropriate
    module based on the current session state and a simple intent
    classifier.  Responses are returned in the format {"module": <module>,
    "text": <assistant_text>}.
    """
    body: Dict[str, Any] = await request.json()
    user_text: str = body.get("text", "")
    forced_module: str = body.get("module")
    session_id: str = body.get("session_id", "default")

    if not user_text:
        return JSONResponse(status_code=400, content={"error": "text field is required"})

    # Initialise session if missing
    session = session_state.setdefault(session_id, {"current_module": "general", "history": []})
    # store the session identifier within the session dict so downstream services
    # can maintain their own state if needed
    session["id"] = session_id

    # Determine the module to use
    if forced_module:
        module = forced_module
    else:
        # If session has an active module other than general, stick with it
        module = session.get("current_module", "general")
        if module == "general":
            # Detect from text
            module = detect_module(user_text)

    # Update the current module in session
    session["current_module"] = module

    # Dispatch to appropriate handler
    if module == "psychology":
        assistant_text = await handle_psychology(session, user_text)
    else:
        assistant_text = await handle_general(session, user_text)

    return {"module": module, "text": assistant_text}


# ----------------------------------------------------------
# Health check and static files
# ----------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount any static frontend (optional).  This code assumes a ``frontend``
# directory exists adjacent to this file; if not, mounting will fail at
# startup.  Comment out the following line if you do not have a
# frontend.
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except Exception:
    logging.warning("Static frontend could not be mounted; directory missing.")


# Global exception handler
@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
