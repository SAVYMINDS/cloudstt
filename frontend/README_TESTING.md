# Testing Enhanced Realtime Storage with Azure Files

## 🎯 What We're Testing

This frontend client will test the enhanced realtime storage system that captures:

1. **Realtime Model Transcripts**: Live updates from the tiny/fast model
2. **Main Model Final Transcript**: High-quality transcription from the main model
3. **Complete Audio Session**: Full audio file saved to Azure Files
4. **Session Metadata**: Configuration, timing, and performance metrics
5. **Azure Files Integration**: Persistent storage that survives container restarts

## 🚀 How to Test

### 1. **Open the Frontend**
```bash
# Navigate to the frontend folder
cd frontend

# Open index.html in your browser
open index.html
# or
python -m http.server 8080  # Then visit http://localhost:8080
```

### 2. **Configure Models**
- **Main Model**: Choose the model for final high-quality transcription (e.g., "base")
- **Realtime Model**: Choose the model for live updates (e.g., "tiny")

### 3. **Test Workflow**
1. **Connect**: Click "Connect" button
   - Watch for session ID and Azure Files storage confirmation
   - Models will load (may take 30-60 seconds for larger models)

2. **Start Recording**: Click "Start Recording"
   - Grant microphone permission
   - Speak clearly for 10-30 seconds
   - Watch realtime transcripts appear

3. **Stop Recording**: Click "Stop Recording"
   - Final transcript will be generated using the main model
   - Complete session data will be saved to Azure Files

## 📊 What to Look For

### ✅ **Success Indicators**

**In the Frontend Logs:**
- `🎯 Session ID: session-xxxxx`
- `🗂️ Azure Files Storage: Enabled and ready`
- `⏳ UPDATING` messages for realtime transcripts
- `🔒 STABILIZED` messages for finalized realtime transcripts
- `🎉 FINAL TRANSCRIPT RECEIVED!`
- `✅ Azure Files Storage: Session data saved successfully!`

**Session Summary Should Show:**
- Session ID
- Total Duration (seconds)
- Audio Chunks count
- Realtime Transcripts count
- Audio URL (file path)

### 🔍 **Server Logs to Monitor**

You can check your container logs for these messages:

```bash
az containerapp logs show --name cloudstt-api-gpu --resource-group mvpv1 --tail 50
```

**Look for:**
- `🗂️ Creating Azure Files storage session for session-xxxxx`
- `✅ Azure Files storage session created successfully`
- `🔒 Realtime stabilized for session-xxxxx`
- `💾 Saving stabilized transcript to Azure Files`
- `🔄 Finalizing Azure Files storage session`
- `✅ Azure Files storage session finalized successfully`
- `JSON saved successfully to Azure Files: /app/azurestorage/...`

## 📁 **Azure Files Structure**

Your files should be saved to:
```
/app/azurestorage/
└── realtime/
    ├── sessions/
    │   └── session-{id}/
    │       └── full_audio.wav
    ├── output/
    │   └── session-{id}/
    │       ├── complete_session_result.json
    │       └── realtime_transcripts/
    │           ├── realtime_transcript_1234567890.json
    │           └── ...
    └── metadata/
        └── session-{id}/
            └── session_info.json
```

## 🐛 **Troubleshooting**

### **Connection Issues**
- Check if the API is running: https://cloudstt-api-gpu.calmhill-33e39416.westus3.azurecontainerapps.io
- Look for WebSocket connection errors in browser console

### **Model Loading Issues**
- Large models (medium/large) may take 2+ minutes to load
- Try smaller models (tiny/base) if you get timeouts
- Check container logs for memory issues

### **Audio Issues**
- Grant microphone permission when prompted
- Speak clearly and avoid background noise
- Check browser console for audio processing errors

### **Storage Issues**
- Check container logs for Azure Files mount status
- Look for "Azure Files mount not found" warnings
- Verify volume mount configuration in container app

## 🎯 **Expected Results**

After a successful test session, you should have:

1. **Complete Session JSON** with both realtime and main model transcripts
2. **Audio File** saved to Azure Files storage
3. **Individual Realtime Transcripts** with timestamps
4. **Session Metadata** with configuration and performance metrics

The final JSON should include:
- `realtime_transcription.transcripts[]` - All live updates
- `main_model_transcription.transcript` - Final high-quality result
- `audio.url` - Path to saved audio file
- `configuration` - All model and session settings
- `metrics` - Performance and timing data

## 📞 **API Endpoint**

The frontend is configured to connect to:
```
wss://cloudstt-api-gpu.calmhill-33e39416.westus3.azurecontainerapps.io/v1/transcribe
```

This is your live production API with Azure Files storage enabled! 