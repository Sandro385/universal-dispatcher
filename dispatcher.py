# dispatcher.py  (backend/dispatcher.py)

import os
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from openai import OpenAI

# OpenAI (Moonshot) კლიენტი – API გასაღები Environment Variable‑ით
client = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv("MOONSHOT_API_KEY")
)

app = FastAPI()

# === Frontend static ფაილები ===
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# === LLM‑ისთვის აღწერილი tools ===
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

# === /chat ===
@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """
    იღებს: {"message": "..."}
    აბრუნებს: {"reply": "..."} ან tool‑ის განლაგების შედეგს.
    """
    body = await request.json()
    user_msg = body.get("message", "")

    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOLS,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        call = tool_calls[0]
        data = json.loads(call.function.arguments)
        return await handle_module(data["module"], data["payload"])

    return JSONResponse({"reply": response.choices[0].message.content})


# === სერვისული გადამისამართება ===
async def handle_module(mod: str, payload: dict) -> JSONResponse:
    """
    დროებითი დამუშავება – უბრალოდ აბრუნებს მიღებულ მონაცემებს.
    აუცილებლობის შემთხვევაში ამ ლოგიკაში ჩაანაცვლეთ თქვენი ბიზნეს‑კოდი.
    """
    return JSONResponse(
        {
            "module": mod,
            "result": f"✅ {mod}-მოდული აღძრულია",
            "payload": payload,
        }
    )

# === /health ===
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

# === ადგილობრივი გაშვება ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
