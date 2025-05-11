from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
import uvicorn

# Initialize service before importing routers
from .service import UnifiedTranscriptionService, job_storage

# Create service instance
transcription_service = UnifiedTranscriptionService()

# Import routers after creating service
from .routes import router as batch_router
from .websocketstt import router as websocket_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Unified Speech Transcription API",
    description="API for batch and real-time speech transcription with speaker diarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include both routers under the /v1 prefix for a Deepgram-style unified endpoint
app.include_router(batch_router, prefix="/v1")
app.include_router(websocket_router, prefix="/v1")

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "online",
        "message": "Unified Speech Transcription API is running",
        "docs": "/docs",
        "websocket_endpoint": "/v1/transcribe",
        "rest_endpoint": "/v1/transcribe"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)