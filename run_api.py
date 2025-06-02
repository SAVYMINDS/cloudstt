import uvicorn
import os
import sys
import multiprocessing
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the start method for multiprocessing (must be done before importing app)
multiprocessing.set_start_method('spawn', force=True)

try:
    logger.info("Starting application import...")
    # Import the FastAPI app
    from api.main import app
    logger.info("Application imported successfully")
except Exception as e:
    logger.error(f"Failed to import application: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("Starting uvicorn server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)