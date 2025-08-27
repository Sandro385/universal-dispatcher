"""
Extended dispatcher service with built‑in registration flow.

This module wraps the original universal‑dispatcher functionality and
introduces a new conversational registration assistant.  The service
switches the user into a registration mode when the user has not yet
registered and has interacted with the assistant beyond a few messages.

Key features:

* **General assistant** – handles everyday questions via an OpenAI model.
* **Psychology assistant** – routes to a dedicated psychological model when
  therapeutic topics are detected via keyword or LLM intent classification.
* **Registration assistant** – automatically prompts the user for basic
  account details (username, email and password) and stores them in a
  simple in‑memory user store.  Once registration is complete, the
  session is marked as registered and subsequent messages are routed back
  to the general assistant.

To run this service you must set the ``OPENAI_API_KEY`` environment
variable.  See README for installation instructions.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List

import bcrypt  # password hashing
from sql_storage import init_db, upsert_user, get_user_hash, add_message, load_history  # sqlite storage helpers

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

###############################################################################
# Configuration and constants
###############################################################################

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env variable is required")

# System prompts define high‑level behaviour for each assistant.
GENERAL_SYSTEM_PROMPT = (
    "შენ ხარ მეგობრული და მრავალმხრივი AI ასისტენტი (Moonshot). "
    "პასუხები უნდა იყოს მოკლე, პროფესიონალური და მომხმარებელზე მორგებული. "
    "გამოიყენე ქართულად საუბარი და საჭიროების შემთხვევაში ტექსტი გადააგზავნე "
    "სათანადო ფორმით."
)

PSYCH_SYSTEM_PROMPT = (
    "შენ ხარ გამოცდილი ფსიქოლოგი. "
    "მომხმარებელს დაეხმარე ემოციურად, უსაფრთხოდ და კონფიდენციალურად. "
    "როცა იგრძნობ, რომ მომხმარებელს აღარ სჭირდება ფსიქოლოგიური დახმარება და "
    "საუბარი გადავიდა ზოგად საკითხებზე, "
    "პასუხის ბოლოს დაამატე სპეციალური მარკერი '[handoff]' (ბრჭყალების გარეშე), "
    "რათა დისპეჩერმა იცოდეს, რომ უნდა დაბრუნდეს ზოგად რეჟიმში."
)

# The registration prompt instructs the model to handle the sign‑up flow.
# Updated to collect only a username and password (email is no longer required).
REGISTRATION_SYSTEM_PROMPT = (
    "შენ ხარ რეგისტრაციის ასისტენტი. "
    "ჯერ ეკითხები მომხმარებელს მომხმარებლის სახელს (username), შემდეგ პაროლს. "
    "თითო ნაბიჯზე ეკითხები მხოლოდ ერთ ველს. "
    "როდესაც ორივე ინფორმაცია შეგროვდება, "
    "დააბრუნე მოკლე ტექსტი \"რეგისტრაცია დასრულებულია\" და JSON:{\"username\":\"...\",\"password\":\"...\"} "
    "და დაამატე მარკერი '[registration_complete]'."
)

# Login prompt for existing users.
LOGIN_SYSTEM_PROMPT = (
    "შენ ხარ ავტორიზაციის ასისტენტი. "
    "ჯერ ეკითხები მომხმარებელს მომხმარებლის სახელს, შემდეგ პაროლს. "
    "როდესაც ორივე მიიღებ, დააბრუნე JSON:{\"username\":\"...\",\"password\":\"...\"} "
    "და დაამატე მარკერი '[login_complete]'."
)

# Regex patterns for detecting psychology and registration intents.
PSYCH_KEYWORDS = re.compile(
    r"ფსიქ|თერაპ|პსიქ|დეპრეს|depress|anxiety|therapy|psych", re.IGNORECASE
)
REGISTRATION_KEYWORDS = re.compile(r"რეგისტ|register|sign\s*up", re.IGNORECASE)

###############################################################################
# Application setup
###############################################################################

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# State management
###############################################################################

# Each session holds current module, history, and registration status.
session_state: Dict[str, Dict[str, Any]] = {}

# Simple in‑memory store for registered users.  Keys are usernames.
users_db: Dict[str, Dict[str, Any]] = {}

# Initialize SQLite database tables.
init_db()

def get_openai_client() -> AsyncOpenAI:
    """Instantiate a new asynchronous OpenAI client."""
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

async def call_openai_chat(messages: List[Dict[str, str]]) -> str:
    """Send messages to the OpenAI chat completion API and return the text."""
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
            # Support streaming responses containing content blocks
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "")
                else:
                    content += getattr(block, "text", "")
        return content
    except Exception as e:
        logging.exception("OpenAI API call failed")
        return f"ხარვეზი მოხდა: {e}"

###############################################################################
# Intent detection
###############################################################################

async def classify_intent_via_llm(text: str) -> float:
    """
    Use the OpenAI model to estimate the probability that a message
    requires psychological support.  Returns a value between 0 and 1.
    """
    classification_prompt = (
        "შენ ხარ ასისტენტი, რომელიც უნდა მიუთითო, რამდენად მაღალია ალბათობა, "
        "რომ მომხმარებლის შეტყობინება საჭიროებს ფსიქოლოგიურ ან თერაპიულ დახმარებას. "
        "იპასუხე მხოლოდ რიცხვით 0-დან 1-მდე, სადაც 0 ნიშნავს – ფსიქოლოგიური თემა არ არის, "
        "და 1 ნიშნავს – ფსიქოლოგიური თემა აშკარად არის."
    )
    messages = [
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": text},
    ]
    response = await call_openai_chat(messages)
    try:
        match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", response)
        if match:
            return float(match.group())
    except Exception:
        pass
    return 0.0

async def detect_module(session: Dict[str, Any], user_text: str) -> str:
    """
    Determine which module should handle the given text.  The order of
    detection is important:

    1. If the session is unregistered and the user has exceeded a
       threshold of interactions, switch to the registration module.
    2. Check explicit registration keywords.
    3. Check for psychology keywords; if none found, use the LLM to
       classify.
    """
    # Detect login intent first
    # If the user explicitly expresses login intent, switch immediately to the
    # login module.  This check must precede the registration threshold check
    # so that login keywords are handled even if the user has sent multiple
    # messages.
    if re.search(r"(შესვლ|login|ავტორიზ|signin)", user_text or "", re.IGNORECASE):
        return "login"

    # If the user is not registered and has already sent at least two
    # messages in the current session, trigger the registration flow.  In
    # earlier versions the threshold was three prior messages (registration on
    # the fourth message) which users found unintuitive.  Lowering the
    # threshold to two prompts registration on the third message instead.
    if not session.get("registered"):
        message_count = sum(1 for m in session.get("history", []) if m["role"] == "user")
        if message_count >= 2:
            return "registration"

    # Explicit registration keywords override other detection.
    if REGISTRATION_KEYWORDS.search(user_text or ""):
        return "registration"

    # Quick keyword check for psychology topics.
    if PSYCH_KEYWORDS.search(user_text or ""):
        return "psychology"

    # Otherwise consult the model to estimate probability.
    probability = await classify_intent_via_llm(user_text or "")
    return "psychology" if probability >= 0.3 else "general"

###############################################################################
# Module handlers
###############################################################################

async def handle_general(session: Dict[str, Any], user_text: str) -> str:
    """
    Handle general conversation with the Moonshot assistant.  Maintains
    conversation history and returns the assistant's reply.
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
    # Keep only the last 40 messages to limit context size
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

async def handle_psychology(session: Dict[str, Any], user_text: str) -> str:
    """
    Handle messages requiring psychological support.  If PSYCH_CHAT_URL is
    configured, forward to the remote service; otherwise use the OpenAI
    model with the psychology system prompt.
    """
    history = session.setdefault("history", [])
    # Check for remote psychology service
    psych_url = os.getenv("PSYCH_CHAT_URL")
    if psych_url:
        try:
            async with httpx.AsyncClient() as client:
                payload = {"text": user_text}
                sid = session.get("id")
                if sid:
                    payload["session_id"] = sid
                resp = await client.post(psych_url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("text", "")
                handoff_flag = data.get("handoff")
                if handoff_flag or "[handoff]" in response_text:
                    # End of psychology session
                    session["current_module"] = "general"
                    response_text = response_text.replace("[handoff]", "").strip()
                return response_text
        except Exception:
            logging.exception("Error calling remote psychology chat service; falling back to local model")

    # Fallback to local model
    messages = [
        {"role": "system", "content": PSYCH_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    if "[handoff]" in response:
        response = response.replace("[handoff]", "").strip()
        session["current_module"] = "general"
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

async def handle_login(session: Dict[str, Any], user_text: str) -> str:
    """
    Guide the user through login using the OpenAI model.  Upon completion,
    verifies credentials and loads conversation history from the database.
    """
    history = session.setdefault("history", [])
    messages = [
        {"role": "system", "content": LOGIN_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]
    response = await call_openai_chat(messages)
    # Update in-memory history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # Check for completion marker
    if "[login_complete]" in response:
        match = re.search(r"\{.*\}", response)
        if match:
            try:
                creds = json.loads(match.group())
                username = creds.get("username")
                password = creds.get("password")
                if username and password:
                    # Verify user credentials
                    pw_hash = get_user_hash(username)
                    if pw_hash and bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8")):
                        # Load conversation history from DB
                        loaded = load_history(username)
                        if loaded:
                            session["history"] = [
                                {"role": r, "content": c} for (r, c) in loaded
                            ]
                        session["registered"] = True
                        session["user"] = username
                        # Clean marker and JSON from response
                        response = re.sub(r"\{.*\}", "", response)
                        response = response.replace("[login_complete]", "").strip()
                        return "შეხვედით სისტემაში. გავაგრძელოთ საუბარი."
                    else:
                        return "სახელი ან პაროლი არასწორია."
            except Exception:
                logging.exception("Failed to parse login JSON")
        # Clean up on failure
        response = re.sub(r"\{.*\}", "", response)
        response = response.replace("[login_complete]", "").strip()
    return response

async def handle_registration(session: Dict[str, Any], user_text: str) -> str:
    """
    Guide the user through registration using the OpenAI model.  Stores
    collected details in users_db once complete and marks the session as
    registered.  All messages in this mode are logged in the session
    history.
    """
    history = session.setdefault("history", [])
    messages = [
        {"role": "system", "content": REGISTRATION_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    # Detect completion marker and extract JSON payload
    if "[registration_complete]" in response:
        # Extract JSON using a simple regex
        match = re.search(r"\{.*\}", response)
        if match:
            try:
                user_info = json.loads(match.group())
                username = user_info.get("username")
                password = user_info.get("password")
                if username and password:
                    # Hash password and upsert user into DB
                    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                    upsert_user(username, pw_hash)
                    # Load any existing history from DB
                    loaded = load_history(username)
                    if loaded:
                        session["history"] = [
                            {"role": r, "content": c} for (r, c) in loaded
                        ]
                    # Mark session as registered and associate user
                    session["registered"] = True
                    session["user"] = username
            except Exception:
                logging.exception("Failed to parse registration JSON")
        # Clean the marker and JSON from the response before returning
        response = re.sub(r"\{.*\}", "", response)
        response = response.replace("[registration_complete]", "").strip()
        # After registration, switch back to general conversation
        session["current_module"] = "general"
    # Limit history size
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

###############################################################################
# Routes
###############################################################################

@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint.  Accepts POST requests with JSON bodies:

        {"text": "...", "module": "...", "session_id": "..."}

    ``text`` is required.  If ``module`` is omitted, the dispatcher
    determines which assistant should handle the message based on the
    conversation context and intent detection.
    """
    body = await request.json()
    user_text = body.get("text", "")
    forced_module = body.get("module")
    session_id = body.get("session_id", "default")

    if not user_text:
        return JSONResponse(status_code=400, content={"error": "text field is required"})

    # Initialise or retrieve session state
    session = session_state.setdefault(session_id, {"current_module": "general", "history": []})
    session["id"] = session_id

    # Persist incoming user message for logged‑in users
    if session.get("user"):
        add_message(session["user"], "user", user_text)

    # Determine module: forced, session or detected
    if forced_module:
        module = forced_module
    else:
        module = session.get("current_module", "general")
        # Only attempt detection if we are currently in general mode
        if module == "general":
            module = await detect_module(session, user_text)

    session["current_module"] = module

    # Dispatch to appropriate handler
    if module == "psychology":
        assistant_text = await handle_psychology(session, user_text)
    elif module == "registration":
        assistant_text = await handle_registration(session, user_text)
    elif module == "login":
        assistant_text = await handle_login(session, user_text)
    else:
        assistant_text = await handle_general(session, user_text)

    # Persist assistant response for logged‑in users
    if session.get("user") and assistant_text:
        add_message(session["user"], "assistant", assistant_text)

    return {"module": module, "text": assistant_text}

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/login")
async def login(request: Request):
    """
    Programmatic login endpoint.  Accepts JSON {"username":..., "password":..., "session_id":...}
    Verifies credentials and loads history into the session.
    """
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    session_id = data.get("session_id")
    if not username or not password or not session_id:
        return {"ok": False, "message": "მონაცემები არასრულია."}
    pw_hash = get_user_hash(username)
    if not pw_hash or not bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8")):
        return {"ok": False, "message": "სახელი ან პაროლი არასწორია."}
    session = session_state.setdefault(session_id, {"current_module": "general", "history": []})
    loaded = load_history(username)
    session["history"] = [
        {"role": r, "content": c} for (r, c) in loaded
    ] if loaded else []
    session["registered"] = True
    session["user"] = username
    return {"ok": True, "message": "შეხვედით სისტემაში."}

# Mount the static frontend if available.  Fail silently if directory is missing.
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except Exception:
    logging.warning("Static frontend could not be mounted; directory missing.")

# Catch‑all exception handler to avoid leaking stack traces to clients
@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})