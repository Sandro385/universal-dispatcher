"""
Dispatcher microservice
=======================

This FastAPI application acts as a simple dispatcher for multiple domain‑specific
LoRA models. It receives a text query along with a target domain and forwards
the request to the appropriate downstream service. The downstream services are
assumed to expose a `/chat` endpoint that accepts a JSON body with a `prompt`
field and returns a JSON response. The dispatcher does not perform any
authentication, registration, or billing logic – those responsibilities are
expected to be handled by higher‑level agents.

Configuration
-------------
The mapping of domain names to service endpoints is defined in the `MODULES`
dictionary. To add a new domain, insert a new key/value pair mapping the
domain identifier to the corresponding base URL (without the path). For
example:

```
MODULES = {
    "psych":  "http://psych-lora:8000/chat",
    "legal":  "http://legal-lora:8000/chat",
    "social": "http://social-lora:8000/chat",
    "finance": "http://finance-lora:8000/chat",  # new domain
}
```

Usage
-----
Run the service with Uvicorn either directly or via Docker. For example:

```
uvicorn dispatcher:app --host 0.0.0.0 --port 8000
```

The service exposes two endpoints:

* `POST /chat` – accepts a JSON body with `text` and optional `domain` fields.
  The `domain` defaults to `psych` if not provided. The dispatcher forwards
  the request to the selected downstream service and returns its JSON
  response.
* `GET /health` – a simple health check returning `{"status": "ok"}`.
"""

import json
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Dispatcher")

# Mapping of domain identifiers to their corresponding LoRA service endpoints.
# You can update these URLs to point to deployed services (e.g. on Render or
# within a Docker network). The default values assume other containers on the
# same network expose a `/chat` endpoint.
MODULES: Dict[str, str] = {
    "psych": "http://psych-lora:8000/chat",
    "legal": "http://legal-lora:8000/chat",
    "social": "http://social-lora:8000/chat",
}


class Query(BaseModel):
    """Model for incoming chat requests."""

    text: str
    domain: str = "psych"


@app.post("/chat")
async def dispatch(query: Query):
    """Forward a chat request to the appropriate domain service.

    Args:
        query: The incoming request containing the text prompt and target domain.

    Returns:
        The JSON response from the downstream service.

    Raises:
        HTTPException: If the specified domain is not configured or if the
        downstream service returns an error.
    """
    domain = query.domain
    if domain not in MODULES:
        raise HTTPException(status_code=404, detail="domain not found")

    # Build the payload expected by the downstream service
    payload = {"prompt": query.text}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(MODULES[domain], json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # If the downstream service returns a non‑2xx status, propagate an error
        raise HTTPException(
            status_code=502,
            detail=f"Downstream service error: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        # Network errors or timeouts
        raise HTTPException(
            status_code=503,
            detail=f"Downstream service unreachable: {exc}",
        ) from exc

    # Return the downstream JSON response directly
    return response.json()


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns a simple JSON response indicating the service is running.
    """
    return {"status": "ok"}