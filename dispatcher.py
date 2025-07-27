# dispatcher.py (entrypoint)

# Import the FastAPI app from backend.dispatcher
from backend.dispatcher import app

if __name__ == "__main__":
    import uvicorn
    # Run the app when this file is executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
