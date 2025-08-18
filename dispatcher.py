# dispatcher.py
import logging
import os
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import openai
from openai import AsyncOpenAI

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env variable is required")

app = FastAPI()

def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------
# Main chat endpoint
# ------------------------------------------------------------------
@app.post("/api/chat")
async def chat(request: Request):
    """
    Example JSON body:
    {
      "module": "psychology",
      "text":   "ვგრძნობ თავს მარტოსულად..."
    }
    """
    body: Dict[str, Any] = await request.json()
    module = body.get("module")
    text   = body.get("text", "")

    if not module or not text:
        return JSONResponse(
            status_code=400,
            content={"error": "module and text are required"},
        )

    return await handle_module(module, {"text": text})

# ------------------------------------------------------------------
# Modules – OpenAI-ს ფსიქოლოგიური მოდული
# ------------------------------------------------------------------
async def handle_module(mod: str, payload: dict) -> dict:
    if mod == "psychology":
        client = get_openai_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "შენ ხარ გამოცდილი ფსიქოლოგი. "
                    "მომხმარებელს დაეხმარე ემოციურად, უსაფრთხოდ და კონფიდენციალურად."
                ),
            },
            {"role": "user", "content": payload["text"]},
        ]

        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
            )

            # Normalize content (string or list of blocks)
            msg = resp.choices[0].message
            content = ""
            if isinstance(msg.content, str):
                content = msg.content
            else:  # list of blocks
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content += block.get("text", "")
                    else:
                        content += getattr(block, "text", "")

            return {
                "module": "psychology",
                "text": content,
            }
        except Exception as e:
            logging.exception("OpenAI psychology module error")
            return {"error": str(e)}

    # დანარჩენი მოდულები — სტუბები
    return {"module": mod, "text": f"სტუბ-პასუხი {mod}-დან"}

# ------------------------------------------------------------------
# Health + UI
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.exception_handler(Exception)
async def catcher(_, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": str(exc)})
