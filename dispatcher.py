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

# უბრალო საკვანძო სიტყვების ფილტრი; პირველ ეტაპზე სწრაფად ამოიცნობს ფსიქოლოგიურ თემებს.
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
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

# Session state: თითოეული session_id‑სთვის ვინახავთ მიმდინარე მოდულს და ისტორიას.
session_state: Dict[str, Dict[str, Any]] = {}

# თუ დაყენებულია, დისპეჩერი ფსიქოლოგიურ შეტყობინებებს გადაუგზავნის ამ URL‑ზე.
PSYCH_CHAT_URL = os.getenv("PSYCH_CHAT_URL")

async def classify_intent_via_llm(text: str) -> float:
    """
    Use an OpenAI model to estimate the probability that a message
    requires psychological support. The model is instructed to reply with a
    number between 0 and 1; 0 = no psychological topic, 1 = clear need.
    """
    classification_prompt = (
        "შენ ხარ ასისტენტი, რომელიც უნდა მიუთითო, რამდენად მაღალია ალბათობა, "
        "რომ მომხმარებლის შეტყობინება საჭიროებს ფსიქოლოგიურ ან თერაპიულ დახმარებას. "
        "Იპასუხე მხოლოდ რიცხვით 0-დან 1-მდე, სადაც 0 ნიშნავს – ფსიქოლოგიური თემა არ არის, "
        "და 1 ნიშნავს – ფსიქოლოგიური თემა აშკარად არის."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": text},
    ]
    response = await call_openai_chat(messages)
    try:
        match = re.search(r"0(?:\\.\\d+)?|1(?:\\.0+)?", response)
        if match:
            return float(match.group())
    except Exception:
        pass
    return 0.0

async def detect_module(text: str) -> str:
    """
    Determine which module should handle the given text.
    1. იყენებს სწრაფ საკვანძო სიტყვების ძებნას.
    2. თუ სიტყვები არ მოიძებნა, იყენებს LLM-ზე დაფუძნებულ კლასიფიკაციას.
    Probability ≥ 0.3 -> „psychology“, სხვაგან -> „general“.
    """
    if PSYCH_KEYWORDS.search(text or ""):
        return "psychology"
    probability = await classify_intent_via_llm(text or "")
    return "psychology" if probability >= 0.3 else "general"

async def call_openai_chat(messages: List[Dict[str, str]]) -> str:
    """Helper to call OpenAI chat API and return the assistant's text."""
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
    """Moonshot მოდულის დამუშავება: ინახავს ისტორიას და აბრუნებს პასუხს."""
    history = session.setdefault("history", [])
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}] + \
               history + [{"role": "user", "content": user_text}]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

async def handle_psychology(session: Dict[str, Any], user_text: str) -> str:
    """
    ფსიქოლოგიური მოდულის დამუშავება. ჯერ ცდება psych-chat სერვერზე
    გადაგზავნას, თუ მითითებულია PSYCH_CHAT_URL; წინააღმდეგ შემთხვევაში
    იყენებს LLM‑ზე დაფუძნებულ ფსიქოლოგიურ პრომპტს. პასუხის ბოლოს
    [handoff] გამოიწვევს დაბრუნებას გენერალურ რეჟიმზე.
    """
    if PSYCH_CHAT_URL:
        try:
            async with httpx.AsyncClient() as client:
                payload = {"text": user_text}
                sid = session.get("id")
                if sid:
                    payload["session_id"] = sid
                resp = await client.post(PSYCH_CHAT_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get("text", "")
                handoff_flag = data.get("handoff")
                if handoff_flag or "[handoff]" in response_text:
                    session["current_module"] = "general"
                    response_text = response_text.replace("[handoff]", "").strip()
                return response_text
        except Exception:
            logging.exception("Error calling remote psychology chat service; falling back to local model")

    history = session.setdefault("history", [])
    messages = [{"role": "system", "content": PSYCH_SYSTEM_PROMPT}] + \
               history + [{"role": "user", "content": user_text}]
    response = await call_openai_chat(messages)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    if "[handoff]" in response:
        response = response.replace("[handoff]", "").strip()
        session["current_module"] = "general"
    if len(history) > 40:
        session["history"] = history[-40:]
    return response

# ----------------------------------------------------------
# Main chat endpoint
# ----------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint to interact with the dispatcher.
    Body: {"text": "...", "module": "...", "session_id": "..."}.
    If module is omitted, dispatcher determines the module itself.
    """
    body = await request.json()
    user_text = body.get("text", "")
    forced_module = body.get("module")
    session_id = body.get("session_id", "default")

    if not user_text:
        return JSONResponse(status_code=400, content={"error": "text field is required"})

    session = session_state.setdefault(session_id, {"current_module": "general", "history": []})
    session["id"] = session_id

    if forced_module:
        module = forced_module
    else:
        module = session.get("current_module", "general")
        if module == "general":
            module = await detect_module(user_text)

    session["current_module"] = module
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

try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except Exception:
    logging.warning("Static frontend could not be mounted; directory missing.")

@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
