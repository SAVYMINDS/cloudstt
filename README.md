# Unified Speech-to-Text API

A production-ready unified API that provides both batch and real-time speech-to-text transcription using OpenAI Whisper models.


## 🚀 Quick Start

### Local Development

1. **Clone and Install**
   ```bash
   git clone <your-repo>
   cd cloudstt
   pip install -r requirements.txt
   ```

2. **Start the API**
   ```bash
   python run_api.py
   ```
   The API will be available at `http://localhost:8000`

3. **View API Documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### Batch Transcription

**Upload Audio File**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "audio_file=@storage/order.wav" \
  -F 'request_data={"mode":"batch","model":"tiny","language":"en","compute_type":"float32"}'
```

**Alternative Batch Request Format**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "audio_file=@audio.mp3" \
  -F 'request_data={"model":"base","language":"auto","compute_type":"float16"}'
```

**Check Job Status**
```bash
curl http://localhost:8000/v1/v1/jobs/{job_id}
```

**Example Success Response:**
```json
{
  "metadata": {
    "request_id": "ad9b6566-9c0a-4d17-adcb-fd321e203fd3",
    "status": "completed",
    "created_at": "2025-06-02T14:56:42.934086",
    "filename": "order.wav",
    "file_size_mb": 21.15,
    "total_processing_time_seconds": 153.2
  },
  "result": {
    "total_duration": 107.967,
    "num_speakers_detected": 2,
    "transcript": "Speaker 1: Thank you for calling Martha's flowers...",
    "segments": [...]
  }
}
```

**Get Job Text Only**
```bash
curl http://localhost:8000/v1/v1/jobs/{job_id}/text
```

**Debug Job Info**
```bash
curl http://localhost:8000/v1/v1/jobs/{job_id}/debug
```

### Real-time Transcription (WebSocket)

```python
import websockets
import json

async def transcribe_realtime():
    async with websockets.connect("ws://localhost:8000/v1/transcribe") as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "config",
            "data": {
                "model": "tiny",
                "language": "en",
                "enable_vad": true
            }
        }))
        
        # Send audio chunks
        await ws.send(json.dumps({
            "type": "audio",
            "data": base64_audio_data
        }))
        
        # Receive transcriptions
        result = await ws.recv()
        print(json.loads(result))
```


### Test WebSocket API

**Option 1: WebSocket Test Frontend (Recommended)**
```bash
# Open the interactive test interface
open frontend/index.html
```

**Frontend Features:**
- 🎤 **Real-time microphone input** with browser audio capture
- 🔧 **Model selection** (tiny, base, small, medium, large) for both realtime and final transcription
- 📊 **Latency metrics** showing model load time and audio processing latency
- 📝 **Live transcript display** with real-time updates and final results
- 🌐 **WebSocket connection management** with auto-reconnect capabilities
- 💾 **Azure Files integration** showing storage session details

**Configuration for Different Environments:**

**Local Development (Current Setup):**
- WebSocket URL: `ws://localhost:8000/v1/transcribe`
- Device: `cpu`
- Compute Type: `float32`

**Azure Container Apps Production:**
To test with Azure, simply edit `frontend/script.js`:

1. **Change the WebSocket URL** (line 2):
   ```javascript
   const WS_URL = "wss://your-app-name.azurecontainerapps.io/v1/transcribe";
   ```

2. **Update the device and compute_type** (around lines 82-83):
   ```javascript
   compute_type: "float16",  // GPU acceleration
   device: "cuda",           // GPU device
   ```

**How to Test:**
1. Ensure your API server is running (`python run_api.py` for local, or deployed to Azure)
2. If testing Azure, edit `frontend/script.js` with your Azure URL and GPU settings (see above)
3. Open `frontend/index.html` in your browser
4. Click **"Connect"** to establish WebSocket connection
5. Select your preferred models (start with "tiny" for fastest response)
6. Click **"Start Recording"** and grant microphone permissions
7. Speak into your microphone and watch real-time transcription
8. Click **"Stop Recording"** to get final transcript with main model
9. Click **"Get Final Transcript"** to retrieve complete session data

**Option 2: Python WebSocket Client**
```bash
# Using the test client script
python tests/test_websocket.py
```

**Option 3: Manual WebSocket Testing**
```python
import websockets
import json
import asyncio

async def test_websocket():
    uri = "ws://localhost:8000/v1/transcribe"
    async with websockets.connect(uri) as websocket:
        # Connect with configuration
        config = {
            "command": "connect",
            "config": {
                "model": "tiny",
                "language": "en",
                "enable_realtime_transcription": True
            }
        }
        await websocket.send(json.dumps(config))
        
        # Listen for responses
        async for message in websocket:
            print(f"Received: {message}")

asyncio.run(test_websocket())
```

## ☁️ Azure Deployment

### Prerequisites
- Azure CLI installed and logged in
- Azure Container Registry
- Azure Container Apps environment

### Deploy to Azure Container Apps

1. **Build and Push Docker Image**
   ```bash
   # Build image
   docker build -t cloudstt:latest .
   
   # Tag for ACR
   docker tag cloudstt:latest yourregistry.azurecr.io/cloudstt:latest
   
   # Push to ACR
   docker push yourregistry.azurecr.io/cloudstt:latest
   ```

2. **Deploy with Azure Files Storage**
   ```bash
   cd azure-deployment
   
   # Setup Azure Files for persistent storage
   ./setup_azure_files.sh
   
   # Deploy the container app
   ./deploy_fixed_storage.sh
   ```

3. **Test Azure Deployment**
   ```bash
   # Get the app URL
   APP_URL=$(az containerapp show --name cloudstt-app --resource-group your-rg --query properties.configuration.ingress.fqdn -o tsv)
   
   # Test batch endpoint
   curl -X POST https://$APP_URL/v1/transcribe \
     -F "audio_file=@test.mp3" \
     -F 'request={"model":"base"}'
   
   # For WebSocket testing: Edit frontend/script.js with your Azure URL
   # Change: const WS_URL = "wss://your-app.azurecontainerapps.io/v1/transcribe"
   # Change: compute_type: "float16", device: "cuda" 
   open frontend/index.html
   ```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
PORT=8000
WORKERS=4
LOG_LEVEL=info

# Model Configuration
DEFAULT_MODEL=base
DEFAULT_COMPUTE_TYPE=float16

# Azure Storage (for Container Apps)
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_SHARE_NAME=cloudstt-storage
```

### Model Options
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39M | Fastest | Good | Real-time, low-latency |
| base | 74M | Fast | Better | Balanced performance |
| small | 244M | Medium | Good | General purpose |
| medium | 769M | Slow | Very Good | High accuracy |
| large | 1550M | Slowest | Best | Maximum accuracy |

## 📁 Project Structure
```