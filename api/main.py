from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app first
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

@app.get("/health")
async def health_check():
    """Health check endpoint for container probes."""
    return {"status": "healthy", "service": "cloudstt-api"}

# Initialize service and routers after basic app setup
try:
    logger.info("Initializing transcription service...")
    from .service import UnifiedTranscriptionService, job_storage
    
    # Create service instance
    transcription_service = UnifiedTranscriptionService()
    logger.info("Transcription service initialized successfully")
    
    # Import routers after creating service
    logger.info("Loading routers...")
    from .routes import router as batch_router
    from .websocketstt import router as websocket_router
    
    # Include both routers under the /v1 prefix for a Deepgram-style unified endpoint
    app.include_router(batch_router, prefix="/v1")
    app.include_router(websocket_router, prefix="/v1")
    logger.info("Routers loaded successfully")
    
except Exception as e:
    error_message = str(e)
    logger.error(f"Failed to initialize services: {error_message}")
    # Don't exit here, let the basic app run for health checks
    
    # Add a fallback endpoint to show the error
    @app.get("/v1/status")
    async def service_status():
        return {
            "status": "error",
            "message": f"Service initialization failed: {error_message}",
            "basic_api": "running"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)