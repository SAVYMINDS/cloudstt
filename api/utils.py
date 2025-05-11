import os
import uuid
from pathlib import Path
from datetime import datetime

def get_temp_file_path(extension: str = "wav") -> str:
    """Generate a temporary file path for downloaded audio files."""
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Clean up old files (optional, can be enabled for production)
    for file in temp_dir.glob("*"):
        if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days > 1:
            try:
                file.unlink()
            except Exception:
                pass
    
    return str(temp_dir / f"{uuid.uuid4()}.{extension}")