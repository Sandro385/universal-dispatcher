import os, json, time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

# ----------------- 1. Moonshot (Kimi – დისპეჩერი) ---------------------------
try:
    from openai import OpenAI              # type: ignore
except ImportError:
    from openai_stub import OpenAI         # type: ignore

moonshot = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)

# ----------------- 2. OpenAI – სპეციალისტური ასისტენტები --------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PSYCHO_ID  = os.getenv("PSYCHO_ASSISTANT_ID")      # asst_…
oai = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ----------------- 3. FastAPI ------------------------------------------------
app = FastAPI()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_to_module",
        "description": "არჩევს შესაბამის მოდულს მოთხოვნის საფუძველზე",
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
}]

# ----------------- 4. Chat endpoint -----------------------------------------
@app.post("/chat")
async def chat(request: Request):
    body      = await request.json()
    user_msg  = body.get("message", "").strip()
    if not moonshot.api_key:
        return {"error": "MOONSHOT_API_KEY is not configured"}

    # 4.1  — პირველი პასუხი Kimi‑სგან
    first = moonshot.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant = first.choices[0].message
    tcalls    = assistant.tool_calls or []

    # 4.2  — ინსტრუმენტი არ დაჭირდა
    if not tcalls:
        return {"reply": assistant.content}

    # 4.3  — ინსტრუმენტი გამოიძახა
    call   = tcalls[0]
    args   = json.loads(call.function.arguments)
    args["payload"].setdefault("text", user_msg)

    module_result = await handle_module(args["module"], args["payload"])

    # 4.4  — tool‑პასუხი და საბოლოო კომენტარი Kimi‑სგან
    follow = moonshot.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": None, "tool_calls": tcalls},
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(module_result, ensure_ascii=False),
            },
        ],
        tool_choice="none",
    )
    return {"reply": follow.choices[0].message.content}

# ----------------- 5. სპეციალისტური მოდულები --------------------------------
async def handle_module(mod: str, payload: dict):
    if mod == "psychology":
        if not (oai and PSYCHO_ID):
            return {"module": "psychology", "result": "error",
                    "detail": "OpenAI creds ან ასისტენტის ID დააკლია"}

        user_text = payload.get("text", "")
        try:
            thread = oai.beta.threads.create()
            oai.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=user_text)
            run = oai.beta.threads.runs.create(
                thread_id=thread.id, assistant_id=PSYCHO_ID)

            while run.status != "completed":
                time.sleep(1)
                run = oai.beta.threads.runs.retrieve(
                    thread_id=thread.id, run_id=run.id)

            msgs = oai.beta.threads.messages.list(thread_id=thread.id)
            reply = msgs.data[0].content[0].text.value.strip() if msgs.data else ""
            return {"module": "psychology", "result": "ok", "advice": reply}

        except Exception as exc:
            return {"module": "psychology", "result": "error", "detail": str(exc)}

    # სხვა მოდულების ნაგულისხმევი პასუხი
    return {"module": mod, "result": f"✅ {mod}-მოდული დასრულდა", "payload": payload}

# ----------------- 6. ჯანმრთელობა & Frontend ---------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":                # ადგილობრივი გაშვებისას
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
