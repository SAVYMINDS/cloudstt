import uvicorn
import os
import sys
import multiprocessing

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the start method for multiprocessing (must be done before importing app)
multiprocessing.set_start_method('spawn', force=True)

# Import the FastAPI app
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)