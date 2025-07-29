import os
import json
import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# --- Moonshot კლინტი (Kimi დისპეჩერი) ---------------------------------------
try:
    from openai import OpenAI                     # type: ignore
except ImportError:                               # fallback development stub
    from openai_stub import OpenAI                # type: ignore

moonshot = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

# --- OpenAI კლინტი ფსიქო‑ასისტენტისთვის -------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PSYCHO_ID  = os.getenv("PSYCHO_ASSISTANT_ID")

oai = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# --- FastAPI -----------------------------------------------------------------
app = FastAPI()

# --- Function‑calling tool – Kimi‑სთვის -------------------------------------
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

# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    """მიღებული შეტყობინების რუტინგი Moonshot‑ის მოდელზე."""
    body = await request.json()
    user_msg = body.get("message", "").strip()

    if not moonshot.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    response = moonshot.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOLS,
        tool_choice="auto",
    )

    tcalls = response.choices[0].message.tool_calls or []
    if tcalls:
        data = json.loads(tcalls[0].function.arguments)
        # ვურთავთ მომხმარებლის ტექსტს payload‑ში, ფსიქო‑მოდულს რომ გამოადგეს
        if "text" not in data["payload"]:
            data["payload"]["text"] = user_msg
        return await handle_module(data["module"], data["payload"])

    return {"reply": response.choices[0].message.content}


# ---------------------------------------------------------------------------
async def handle_module(mod: str, payload: dict):
    """სპეციალისტური მოდულების დამუშავება."""
    # --- ფსიქოლოგიური მოდული --------------------------------------------------
    if mod == "psychology":
        if not (oai and PSYCHO_ID):
            return {
                "module": "psychology",
                "result": "error",
                "detail": "OpenAI client or Assistant ID not configured",
            }

        user_text = payload.get("text", "")
        try:
            # ① Thread‑ის შექმნა
            thread = oai.beta.threads.create()
            oai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_text,
            )

            # ② ასისტენტის გაშვება
            run = oai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=PSYCHO_ID,
            )

            # ③ Poll‑ing – ვცდით დასრულებამდე
            while run.status != "completed":
                time.sleep(1)
                run = oai.beta.threads.runs.retrieve(
                    thread_id=thread.id, run_id=run.id
                )

            # ④ ბოლო პასუხის ამოღება
            msgs = oai.beta.threads.messages.list(thread_id=thread.id)
            assistant_reply = (
                msgs.data[0].content[0].text.value.strip() if msgs.data else ""
            )

            return {
                "module": "psychology",
                "result": "ok",
                "advice": assistant_reply,
            }
        except Exception as exc:
            return {
                "module": "psychology",
                "result": "error",
                "detail": str(exc),
            }

    # --- სხვა მოდულები (სტუბი) -------------------------------------------------
    return {
        "module": mod,
        "result": f"✅ {mod}-მოდული შესრულდა წარმატებით",
        "payload": payload,
    }


# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# Static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
