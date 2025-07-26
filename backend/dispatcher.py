# backend/dispatcher.py
import os, json, httpx, re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODULES = {
    "psych":  "http://psych-lora:8000/chat",
    "legal":  "http://legal-lora:8000/chat",
    "social": "http://social-lora:8000/chat"
}

class Query(BaseModel):
    text: str
    domain: str = "psych"

@app.post("/chat")
async def dispatch(q: Query):
    if q.domain not in MODULES:
        raise HTTPException(status_code=404, detail="domain not found")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(MODULES[q.domain], json={"prompt": q.text})
    return {"reply": resp.json().get("reply", "")}

@app.get("/health")
def health():
    return {"status": "ok"}
