"""
Dispatcher service responsible for routing user messages to different
specialised assistants. In addition to the original functionality of
forwarding to a psychological assistant when explicitly requested via
the ``module`` field, this version keeps track of per‑session
conversation state and can automatically switch between a general
assistant (e.g. Moonshot) and a psychological assistant based on the
user's intent. When the user engages in therapeutic topics the
conversation is delegated to the psychology module until that module
signals that it is ready to hand control back to the general assistant.

Session state is tracked in memory using a simple dictionary keyed by
``session_id`` values supplied in the request body. Each entry
contains the current active module and the running chat history for
that session. A very simple keyword‑based classifier is used to
determine when a message should be routed to the psychology module if
no explicit module is provided. For production deployments you may
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# გაუმჯობესებული საკვანძო სიტყვების ფილტრი
PSYCH_KEYWORDS = re.compile(
    r"ფსიქოლოგ|თერაპი|დეპრეს|სტრეს|წყენ|მწუხარ|anxiety|depression|therapy|psych|მენტალურ|ემოციურ|"
    r"სამხნევ|უიმედობ|შიშ|ფობია|პანიკ|ტრავმ|გულისტკივ", 
    re.IGNORECASE
)

app = FastAPI(title="Chat Dispatcher", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_openai_client() -> AsyncOpenAI:
    """Get configured OpenAI client"""
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

# Session state: თითოეული session_id‑სთვის ვინახავთ მიმდინარე მოდულს და ისტორიას.
session_state: Dict[str, Dict[str, Any]] = {}

# თუ დაყენებულია, დისპეჩერი ფსიქოლოგიურ შეტყობინებებს გადაუგზავნის ამ URL‑ზე.
PSYCH_CHAT_URL = os.getenv("PSYCH_CHAT_URL")
logger.info(f"PSYCH_CHAT_URL configured: {bool(PSYCH_CHAT_URL)}")

async def classify_intent_via_llm(text: str) -> float:
    """
    Use an OpenAI model to estimate the probability that a message
    requires psychological support. The model is instructed to reply with a
    number between 0 and 1; 0 = no psychological topic, 1 = clear need.
    """
    classification_prompt = (
        "შენ ხარ ასისტენტი, რომელიც უნდა მიუთითო, რამდენად მაღალია ალბათობა, "
        "რომ მომხმარებლის შეტყობინება საჭიროებს ფსიქოლოგიურ ან თერაპიულ დახმარებას. "
        "პასუხი უნდა იყოს მხოლოდ რიცხვი 0-დან 1-მდე (მაგ: 0.7), სადაც 0 ნიშნავს – ფსიქოლოგიური თემა არ არის, "
        "და 1 ნიშნავს – ფსიქოლოგიური თემა აშკარად არის."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": text},
    ]
    
    try:
        response = await call_openai_chat(messages)
        logger.debug(f"LLM classification response: {response}")
        
        # გამოსწორებული regex - ორმაგი escaping წაშლილია
        match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", response)
        if match:
            value = float(match.group())
            logger.debug(f"Extracted probability: {value}")
            return value
        
        # ალტერნატიულად, ვეძებთ ნებისმიერ რიცხვს 0-1 შორის
        number_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response)
        if number_match:
            value = float(number_match.group(1))
            # 0-1 დიაპაზონში შემოვიყვანოთ
            value = max(0.0, min(1.0, value))
            logger.debug(f"Alternative extracted probability: {value}")
            return value
            
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
    
    return 0.0

async def detect_module(text: str, current_module: str = "general") -> str:
    """
    განსაზღვრავს რომელი მოდული უნდა დაამუშავოს მოცემული ტექსტი.
    ყურადღება აქვს მიმდინარე მოდულსაც context-ისთვის.
    """
    if not text:
        return current_module
    
    # სწრაფი საკვანძო სიტყვების ძებნა
    if PSYCH_KEYWORDS.search(text):
        logger.info(f"Psychology keywords detected in: {text[:50]}...")
        return "psychology"
    
    # LLM-ზე დაფუძნებული კლასიფიკაცია
    try:
        probability = await classify_intent_via_llm(text)
        logger.info(f"LLM classification probability: {probability} for text: {text[:50]}...")
        
        # psychology მოდულისთვის threshold - თუ უკვე psychology რეჟიმშია, უფრო ადვილად რჩება
        threshold = 0.4 if current_module == "general" else 0.3
        
        if probability >= threshold:
            return "psychology"
            
    except Exception as e:
        logger.error(f"Module detection failed: {e}")
    
    return current_module if current_module == "psychology" else "general"

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
            # Handle structured content
            for block in msg.content or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "")
                else:
                    content += getattr(block, "text", "")
        return content.strip()
    except Exception as e:
        logger.exception("OpenAI API call failed")
        return f"ხარვეზი მოხდა: {e}"

async def handle_general(session: Dict[str, Any], user_text: str) -> str:
    """Moonshot მოდულის დამუშავება: ინახავს ისტორიას და აბრუნებს პასუხს."""
    history = session.setdefault("history", [])
    
    # Build messages for API call
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}] + \
               history + [{"role": "user", "content": user_text}]
    
    logger.debug(f"General module processing: {len(messages)} messages")
    response = await call_openai_chat(messages)
    
    # Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    
    # Keep history manageable
    if len(history) > 40:
        session["history"] = history[-40:]
        logger.debug("History trimmed to last 40 messages")
    
    return response

async def handle_psychology(session: Dict[str, Any], user_text: str) -> str:
    """
    ფსიქოლოგიური მოდულის დამუშავება. ჯერ ცდება psych-chat სერვერზე
    გადაგზავნას, თუ მითითებულია PSYCH_CHAT_URL; წინააღმდეგ შემთხვევაში
    იყენებს LLM‑ზე დაფუძნებულ ფსიქოლოგიურ პრომპტს. პასუხის ბოლოს
    [handoff] გამოიწვევს დაბრუნებას გენერალურ რეჟიმზე.
    """
    # Try external psychology service first if configured
    if PSYCH_CHAT_URL:
        try:
            async with httpx.AsyncClient() as client:
                payload = {"text": user_text}
                sid = session.get("id")
                if sid:
                    payload["session_id"] = sid
                
                logger.info(f"Calling external psychology service: {PSYCH_CHAT_URL}")
                resp = await client.post(PSYCH_CHAT_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                response_text = data.get("text", "")
                handoff_flag = data.get("handoff", False)
                
                # Check for handoff signal
                if handoff_flag or "[handoff]" in response_text:
                    logger.info("Psychology module signaled handoff to general")
                    session["current_module"] = "general"
                    response_text = response_text.replace("[handoff]", "").strip()
                
                return response_text
                
        except Exception as e:
            logger.exception(f"Error calling remote psychology service at {PSYCH_CHAT_URL}")
            logger.info("Falling back to local psychology model")

    # Fallback to local psychology model
    history = session.setdefault("history", [])
    
    # Build messages for API call
    messages = [{"role": "system", "content": PSYCH_SYSTEM_PROMPT}] + \
               history + [{"role": "user", "content": user_text}]
    
    logger.debug(f"Psychology module (local) processing: {len(messages)} messages")
    response = await call_openai_chat(messages)
    
    # Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": response})
    
    # Check for handoff signal
    if "[handoff]" in response:
        logger.info("Local psychology model signaled handoff to general")
        response = response.replace("[handoff]", "").strip()
        session["current_module"] = "general"
    
    # Keep history manageable
    if len(history) > 40:
        session["history"] = history[-40:]
        logger.debug("History trimmed to last 40 messages")
    
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
    try:
        body = await request.json()
        user_text = body.get("text", "").strip()
        forced_module = body.get("module")
        session_id = body.get("session_id", "default")

        if not user_text:
            return JSONResponse(
                status_code=400, 
                content={"error": "text field is required and cannot be empty"}
            )

        logger.info(f"Chat request - Session: {session_id}, Text: {user_text[:100]}...")

        # Initialize or get session state
        session = session_state.setdefault(session_id, {
            "current_module": "general", 
            "history": [],
            "id": session_id
        })

        current_module = session.get("current_module", "general")
        logger.info(f"Session {session_id}: Current module: {current_module}")

        # Determine target module
        if forced_module:
            module = forced_module
            logger.info(f"Module forced to: {module}")
        else:
            module = await detect_module(user_text, current_module)
            logger.info(f"Module detected as: {module} (was: {current_module})")

        # Update session state
        session["current_module"] = module

        # Process message with appropriate handler
        if module == "psychology":
            assistant_text = await handle_psychology(session, user_text)
        else:
            assistant_text = await handle_general(session, user_text)

        # Log response info
        final_module = session.get("current_module", module)
        logger.info(f"Response generated by {module} module, final module: {final_module}")
        logger.debug(f"Response: {assistant_text[:200]}...")

        return {
            "module": final_module, 
            "text": assistant_text,
            "session_id": session_id
        }

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )

# ----------------------------------------------------------
# Additional endpoints
# ----------------------------------------------------------
@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session"""
    session = session_state.get(session_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    
    return {
        "session_id": session_id,
        "current_module": session.get("current_module"),
        "history_length": len(session.get("history", [])),
        "last_messages": session.get("history", [])[-5:] if session.get("history") else []
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session"""
    if session_id in session_state:
        del session_state[session_id]
        return {"message": f"Session {session_id} cleared"}
    return JSONResponse(status_code=404, content={"error": "Session not found"})

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(session_state.keys()),
        "total": len(session_state)
    }

# ----------------------------------------------------------
# Health check and static files
# ----------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "psych_service_configured": bool(PSYCH_CHAT_URL),
        "active_sessions": len(session_state)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chat Dispatcher Service",
        "endpoints": ["/chat", "/health", "/sessions", "/session/{id}"],
        "version": "1.0.0"
    }

# Mount static files if available
try:
    app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")
    logger.info("Static files mounted from 'frontend' directory")
except Exception as e:
    logger.warning(f"Static frontend could not be mounted: {e}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.exception("Unhandled error occurred")
    return JSONResponse(
        status_code=500, 
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
