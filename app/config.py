"""
Configuration module for audio transcription services.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Base configuration class."""
    # Basic app config
    DEBUG = False
    TESTING = False
    
    # Audio processing parameters
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHUNK_SIZE = 16000  # 1 second chunks at 16kHz
    
    # Transcription parameters
    TRANSCRIPTION_MODEL = "medium"  # base, small, medium, large-v2
    ALLOWED_LATENCY_LIMIT = 1700
    
    # Storage paths
    STORAGE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "storage"
    )
    
    # Azure Storage configuration
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    
    # Container names
    INPUT_CONTAINER = "audio-input"
    PROCESSING_CONTAINER = "audio-processing"
    OUTPUT_CONTAINER = "audio-output"
    METADATA_CONTAINER = "audio-metadata"


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    # Use larger models for better accuracy in production
    TRANSCRIPTION_MODEL = "medium"
    
    # Production typically has more resources
    ALLOWED_LATENCY_LIMIT = 3000


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    # Use small/fast models for testing
    TRANSCRIPTION_MODEL = "small"


def get_config() -> Dict[str, Any]:
    """
    Get configuration as a dictionary based on environment.
    
    Returns:
        Configuration dictionary
    """
    env = os.environ.get("ENVIRONMENT", "development").lower()
    
    if env == "production":
        config_class = ProductionConfig
    elif env == "testing":
        config_class = TestingConfig
    else:
        config_class = DevelopmentConfig
    
    # Convert class attributes to dictionary
    return {key: value for key, value in config_class.__dict__.items() 
            if not key.startswith('__')}