# dispatcher.py (entrypoint)

import os
import sys

# Ensure the backend package is found relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from backend.dispatcher import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
