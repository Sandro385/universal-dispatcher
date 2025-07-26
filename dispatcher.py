"""
Dispatcher microservice
=======================

This FastAPI application acts as a simple dispatcher for multiple
domain‑specific LoRA models.  It receives a text query along with a
target domain and forwards the request to the appropriate downstream
service.  The downstream services are assumed to expose a `/chat`
endpoint that accepts a JSON body with a `prompt` field and returns
a JSON response with a `reply` key.  The dispatcher performs
basic validation of the domain and proxies the request using
``httpx.AsyncClient``.

The API supports CORS so that a web front‑end hosted on a different
origin can make requests directly to the dispatcher.  See the
``frontend/index.html`` file for an example terminal‑style UI.

Usage
-----
Run the dispatcher with Uvicorn either directly or via Docker.  For
example:

```
uvicorn dispatcher:app --host 0.0.0.0 --port 8000
```

When using the Dockerfile provided in this repository, the ``CMD``
in the Dockerfile starts the dispatcher automatically.

Endpoints
---------

* ``POST /chat`` – accepts a JSON body with ``text`` and an optional
  ``domain`` field.  The domain defaults to ``psych`` if not provided.
  The dispatcher forwards the request to the selected downstream
  service and returns an object ``{"reply": "..."}`` containing the
  response from the downstream service.  If the domain is not
  configured, a 404 error is returned.
* ``GET /health`` – a simple health check returning ``{"status": "ok"}``.
"""

import json
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create the FastAPI application and enable CORS for all origins.  In
# a production environment you may wish to restrict this to specific
# front‑end domains.
app = FastAPI(title="Dispatcher")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapping of domain identifiers to their corresponding LoRA service
# endpoints.  These should be updated to reflect the actual URLs of
# the deployed LoRA models.  When running via docker-compose, the
# service names correspond to the compose service names defined in
# ``docker-compose.yml``.  When deployed on Render, you would set
# these to the external URLs of your LoRA services.
MODULES: Dict[str, str] = {
    "psych": "http://psych-lora:8000/chat",
    "legal": "http://legal-lora:8000/chat",
    "social": "http://social-lora:8000/chat",
}


class Query(BaseModel):
    """Schema for incoming chat requests."""

    text: str
    domain: str = "psych"


@app.post("/chat")
async def dispatch(q: Query):
    """Forward a chat request to the appropriate domain service.

    Args:
        q: The incoming request containing the text prompt and target domain.

    Returns:
        A dict with a ``reply`` key containing the response from the downstream service.

    Raises:
        HTTPException: If the specified domain is not configured or if the
        downstream service returns an error.
    """
    # Validate the requested domain
    if q.domain not in MODULES:
        raise HTTPException(status_code=404, detail="domain not found")

    # Construct the payload expected by the downstream service.
    # Our LoRA services expect a 'prompt' field.
    payload = {"prompt": q.text}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(MODULES[q.domain], json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Downstream service error: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Downstream service unreachable: {exc}",
        ) from exc

    # Extract the reply from the downstream service.  Downstream is expected
    # to return a JSON object with a 'reply' key.  If not present, return
    # an empty string.
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON response from service")
    reply = data.get("reply", "")
    return {"reply": reply}


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns a simple JSON response indicating the service is running.
    """
    return {"status": "ok"}