# Universal Dispatcher

A simple FastAPI microservice that forwards chat requests to domain‑specific
LoRA services. This dispatcher acts as a central entry point for various
specialised language models (e.g. *psych*, *legal*, *social*), routing each
incoming prompt to the appropriate service based on the supplied domain.

## Features

* **Single API**: Exposes a single `/chat` endpoint for clients to send prompts
  along with a `domain` field indicating the target model.
* **Domain routing**: Looks up the requested domain in a configurable mapping
  and forwards the request to the corresponding `/chat` endpoint of a
  downstream service.
* **Health check**: Provides a `/health` endpoint returning `{ "status": "ok" }`.
* **Extensible**: Add more domains by extending the `MODULES` dictionary in
  `dispatcher.py`.

## Project structure

```
universal_dispatcher/
├── dispatcher.py      # FastAPI app implementing the dispatcher
├── frontend/          # Static terminal‑style UI for interacting with the dispatcher
│   └── index.html     # Simple JavaScript/HTML terminal interface
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition for deployment
└── README.md          # This file
```

## Installation and running locally

1. Clone the repository and change into the project directory:

   ```bash
   git clone https://github.com/<your-username>/universal_dispatcher.git
   cd universal_dispatcher
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --no-cache-dir -r requirements.txt
   ```

3. Start the service using Uvicorn:

   ```bash
   uvicorn dispatcher:app --host 0.0.0.0 --port 8000
   ```

4. Send a request to the dispatcher:

   ```bash
   curl -X POST http://localhost:8000/chat \
        -H 'Content-Type: application/json' \
        -d '{"text": "Hello, world", "domain": "psych"}'
   ```

## Docker

A `Dockerfile` is provided to build a container image. To build and run the
container locally:

```bash
docker build -t universal-dispatcher .
docker run -p 8000:8000 universal-dispatcher
```

On platforms like [Render](https://render.com), the `CMD` in the Dockerfile
uses the `PORT` environment variable provided by the platform.

## Deployment on Render

1. Create a new **Web Service** on Render and connect it to your GitHub
   repository containing this project.
2. Leave the build and start commands blank – Render will detect the
   `Dockerfile` and build the container automatically.
3. Set the **Health Check Path** to `/health` so Render knows when the
   service is ready.
4. Deploy using the **Free** instance type (unless your usage requires more
   resources).

## Front‑end

This repository includes a minimal web front‑end under `frontend/index.html`.
It renders a black “terminal” with a blinking cursor.  Users can type text
and press `Enter` to send it to the dispatcher, and the response from the
selected domain is displayed below.  By default the JavaScript points to
`https://universal-dispatcher.onrender.com/chat`.  When running locally,
you can change the `api` constant in the file to `http://localhost:8000/chat`.

To serve the front‑end locally you can run a simple static file server from
the `frontend` directory:

```bash
python -m http.server 8080 --directory frontend
```

Then open `http://localhost:8080/index.html` in your browser.

## Limitations

This dispatcher does not implement authentication, registration or billing. It
is intended to be a thin routing layer. A separate “admin” agent or service
should handle user management and payment flows.