import os
import json
import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 1) Moonshot‑კლაიენტი (Kimi — დისპეჩერი)
try:
    from openai import OpenAI                   # type: ignore
except ImportError:                             # development fallback
    from openai_stub import OpenAI              # type: ignore

moonshot = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

# ---------------------------------------------------------------------------
# 2) OpenAI‑კლაიენტი სპეციალური ასისტენტებისთვის (ფსიქო‑და სხვ.)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PSYCHO_ID  = os.getenv("PSYCHO_ASSISTANT_ID")       # asst_########

oai = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------------------------------------------------------------------------
app = FastAPI()

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
    """ხელმძღვანელობს თხოვნებს Moonshot‑ის მოდელთან და საჭიროებისას სპეციალისტურ მოდულებს რთავს."""
    body = await request.json()
    user_msg = body.get("message", "").strip()

    if not moonshot.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # ① პირველი გამოძახება — Kimi არჩევს, სჭირდება თუ არა ინსტრუმენტი
    first = moonshot.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant = first.choices[0].message
    tcalls = assistant.tool_calls or []

    # --- ა) ინსტრუმენტს არ გამოუძახებია → პირდაპირ ტექსტი ---
    if not tcalls:
        return {"reply": assistant.content}

    # --- ბ) ინსტრუმენტი გამოიძახა → შიდა მოდულის შესრულება ---
    call      = tcalls[0]
    arguments = json.loads(call.function.arguments)
    if "text" not in arguments["payload"]:
        arguments["payload"]["text"] = user_msg   # გადავაწოდოთ თვითონ ტექსტიც

    module_result = await handle_module(arguments["module"], arguments["payload"])

    # ② ვაგზავნით tool‑პასუხს zurück Moonshot‑ში, რათა საბოლოო ტექსტი მოახდინოს
    follow_msgs = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": None, "tool_calls": tcalls},
        {
            "role": "tool",
            "tool_call_id": call.id,              # აუცილებლად იგივე ID
            "name": call.function.name,
            "content": json.dumps(module_result, ensure_ascii=False),
        },
    ]

    final = moonshot.chat.completions.create(
        model="moonshot-v1-8k",
        messages=follow_msgs,
        tool_choice="none",                       # აღარ გამოვიძახებთ ახალ ინსტრუმენტს
    )

    return {"reply": final.choices[0].message.content}


# ---------------------------------------------------------------------------
async def handle_module(mod: str, payload: dict):
    """სპეციალისტური მოდულების რეალიზაცია."""
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

            # ③ Poll‑ing დასრულებამდე
            while run.status != "completed":
                time.sleep(1)
                run = oai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            # ④ ბოლო პასუხის ამოღება
            msgs = oai.beta.threads.messages.list(thread_id=thread.id)
            assistant_reply = msgs.data[0].content[0].text.value.strip() if msgs.data else ""

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

    # --- სხვა მოდულების სტუბი --------------------------------------------------
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
