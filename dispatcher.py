 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/dispatcher.py b/dispatcher.py
index c597bc2f246c6429a88b2520a596b377305256b9..271197edf3452aff188e300d19c9f5fa17ffcd00 100644
--- a/dispatcher.py
+++ b/dispatcher.py
@@ -162,49 +162,67 @@ async def chat(request: Request):
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
+
+            # Newer OpenAI models may return ``message.content`` either as a
+            # plain string or as a list of "content blocks".  Normalize the
+            # response so the rest of the dispatcher always receives a simple
+            # text string.
+            msg = resp.choices[0].message
+            content = ""
+            if isinstance(msg.content, str):
+                content = msg.content
+            else:  # list of blocks
+                for block in msg.content:
+                    if isinstance(block, dict):
+                        if block.get("type") == "text":
+                            content += block.get("text", "")
+                    else:
+                        text = getattr(block, "text", "")
+                        content += text
+
             return {
                 "module": "psychology",
-                "text": resp.choices[0].message.content,
+                "text": content,
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
 
EOF
)
