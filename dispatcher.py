import os
import json
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from openai import OpenAI          # type: ignore
except ImportError:
    from openai_stub import OpenAI     # type: ignore

# --- გარემოს ცვლადები ---------------------------------------------------------
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
MOONSHOT_API_KEY  = os.getenv("MOONSHOT_API_KEY")

# --- OpenAI/Moonshot კლიენტი --------------------------------------------------
client = OpenAI(
    base_url=MOONSHOT_BASE_URL,
    api_key=MOONSHOT_API_KEY,
)

MODEL = "kimi-k2-0711-preview"

# --- FastAPI ------------------------------------------------------------------
app = FastAPI()

# --- ხელმისაწვდომი ინსტრუმენტი (მოდულების როუტერი) --------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "route_to_module",
            "description": (
                "არჩევს შესაბამის მოდულს, "
                "მაგალითად ფსიქოლოგიურ, იურიდიულ, FAQ ან fallback-ს. "
                "გამოიყენე მხოლოდ მაშინ, თუ უშუალო პასუხი საკმარისი არ არის."
            ),
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

# --- System prompt ------------------------------------------------------------
SYSTEM_MSG = {
    "role": "system",
    "content": (
        "შენ ხარ მრავალპროფილიანი, პროფესიონალური ასისტენტი. "
        "უპასუხე აბონენტის კითხვას პირდაპირ. "
        "მხოლოდ მაშინ გამოიყენე ფუნქცია route_to_module, "
        "თუ სპეციალისტის ჩართვა ნამდვილად აუცილებელია."
    ),
}

# ------------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    """
    POST /chat
    Body: { "message": "<user text>" }
    Returns: { "reply": "<assistant reply>" }
    """
    body = await request.json()
    user_msg: str = body.get("message", "").strip()

    if not user_msg:
        return {"error": "empty message"}
    if not client.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # -------- პირველადი გამოძახება — tools გარეშე ----------------------------
    messages: List[Dict[str, str]] = [
        SYSTEM_MSG,
        {"role": "user", "content": user_msg},
    ]

    try:
        first = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            # TOOLS მაინც გადავცემთ, რათა მოდელმა ფუნქცია *შეეძლოს* გამოიძახოს
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.5,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    assistant_msg = first.choices[0].message
    tcalls = assistant_msg.tool_calls or []

    # --- თუ ფუნქცია არ გამოუძახია, პირდაპირ ვაბრუნებთ პასუხს -------------------
    if not tcalls:
        return {"reply": assistant_msg.content}

    # -------- ფუნქცია გამოიძახა → სერვერზე ვასრულებთ --------------------------
    try:
        # ამ მაგალითში ვიღებთ მხოლოდ პირველ call-ს
        call = tcalls[0]
        args = json.loads(call.function.arguments)
        module_result = await handle_module(args["module"], args["payload"])
    except Exception as exc:
        return {"error": f"Module error: {type(exc).__name__}: {exc}"}

    # -------- ფუნქციის შედეგი ვაბრუნებთ მოდელს, რომ „გადაიღებოს“ -------------
    messages.extend([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tcalls,
        },
        {
            "role": "function",
            "name": call.function.name,
            "content": json.dumps(module_result, ensure_ascii=False),
        },
    ])

    try:
        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tool_choice="none",  # აქ აღარ ვრთავთ ახალ ფუნქციებს
            temperature=0.8,
            top_p=0.95,
            presence_penalty=0.5,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    final_reply = final.choices[0].message.content or ""
    return {"reply": final_reply}

# ------------------------------------------------------------------------------
async def handle_module(mod: str, payload: dict) -> dict:
    """
    შიდა ფუნქცია, რომელიც ასრულებს კონკრეტულ მოდულს და აბრუნებს შედეგს.
    """
    # --- აქ შეგიძლიათ რეალური ბიზნეს‑ლოგიკა ჩასვათ ---------------------------
    return {
        "module": mod,
        "result": f"✅ {mod}-მოდული შესრულდა წარმატებით",
        "payload": payload,
    }

# ------------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Static Frontend (Optional) ----------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# --- გლობალური შეცდომათა დამჭერი ---------------------------------------------
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}"}
    )
