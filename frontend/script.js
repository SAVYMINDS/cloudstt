// WebSocket Server URL
const WS_URL = "ws://localhost:8000/v1/transcribe";

// Configuration
const CONFIG = {
    connectionTimeout: 60000,  // 1 minute timeout for connection
    modelLoadTimeout: 120000   // 2 minutes timeout for model loading
};

// DOM Elements
const connectButton = document.getElementById('connectButton');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusMessage = document.getElementById('statusMessage');
const serverMessages = document.getElementById('serverMessages');
const realtimeTranscript = document.getElementById('realtimeTranscript');
const finalTranscript = document.getElementById('finalTranscript');
const mainModelSelect = document.getElementById('mainModel');
const realtimeModelSelect = document.getElementById('realtimeModel');

// WebSocket object
let websocket = null;

// Audio context and processor
let audioContext = null;
let mediaStreamSource = null;
const TARGET_SAMPLE_RATE = 16000;

// Latency measurement variables
let connectionStartTime;
let modelReadyTime;
let firstAudioChunkTime;
let firstTranscriptionTime;
let isMeasuringFirstChunk = true;
let latencyStats = {
    modelLoadTimes: [],
    processingLatencies: []
};

// Event Listeners
connectButton.addEventListener('click', () => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        logServerMessage("Already connected.");
        return;
    }

    // Reset latency measurement for this connection
    connectionStartTime = performance.now();
    firstTranscriptionTime = null;
    isMeasuringFirstChunk = true;
    
    updateStatus("Connecting...");
    logServerMessage(`Attempting to connect to ${WS_URL}`);
    connectButton.disabled = true;

    // Set connection timeout
    const connectionTimeoutId = setTimeout(() => {
        if (websocket && websocket.readyState === WebSocket.CONNECTING) {
            logServerMessage("Connection timeout - WebSocket connection taking too long. Please try again.");
            websocket.close();
            connectButton.disabled = false;
            updateStatus("Connection Timeout");
        }
    }, CONFIG.connectionTimeout);

    websocket = new WebSocket(WS_URL);

    websocket.onopen = (event) => {
        // Clear connection timeout
        clearTimeout(connectionTimeoutId);
        
        updateStatus("Connected");
        logServerMessage("Successfully connected to WebSocket server.");
        startButton.disabled = false;
        stopButton.disabled = true;
        
        // Get selected model configurations
        const mainModel = mainModelSelect.value;
        const realtimeModel = realtimeModelSelect.value;
        
        // Special handling for larger models
        if (mainModel.includes("medium") || mainModel.includes("large") || 
            realtimeModel.includes("medium") || realtimeModel.includes("large")) {
            logServerMessage("You've selected a larger model (medium/large). Loading may take longer and might require more GPU memory.");
        }
        
        // Set model load timeout (for waiting for 'connected' event)
        const modelLoadTimeoutId = setTimeout(() => {
            if (!modelReadyTime) {
                logServerMessage("Model load timeout - The server is taking too long to load the model. Consider using a smaller model or check server resources.");
                updateStatus("Model Load Timeout");
            }
        }, CONFIG.modelLoadTimeout);
        
        // Send the initial 'connect' command to the server with enhanced model configurations
        const connectPayload = {
            command: "connect",
            config: {
                // Enhanced configuration for realtime storage testing
                realtime_model_type: realtimeModel,
                model: mainModel,
                    language: "en",
                compute_type: "float32",
                device: "cpu",
                
                // Realtime settings
                enable_realtime_transcription: true,
                use_main_model_for_realtime: false,
                realtime_processing_pause: 0.2,
                
                // VAD settings
                silero_sensitivity: 0.4,
                webrtc_sensitivity: 3,
                post_speech_silence_duration: 0.6,
                min_length_of_recording: 0.5,
                
                // Text formatting
                ensure_sentence_starting_uppercase: true,
                ensure_sentence_ends_with_period: true
            }
        };
        websocket.send(JSON.stringify(connectPayload));
        logServerMessage("Sent 'connect' command to server with model configurations.");
    };

    websocket.onmessage = (event) => {
        try {
            const messageData = JSON.parse(event.data);
            logServerMessage(`Received: ${JSON.stringify(messageData)}`);

            // Handle different event types from server
            if (messageData.event === "connected") {
                // Server confirmed connection and initialization
                logServerMessage(`🎯 Session ID: ${messageData.session_id}`);
                logServerMessage(`🤖 Realtime Model: ${realtimeModelSelect.value}`);
                logServerMessage(`🧠 Main Model: ${mainModelSelect.value}`);
                logServerMessage(`🗂️ Azure Files Storage: Enabled and ready`);
                
                // Calculate and log model loading time
                modelReadyTime = performance.now();
                const modelLoadTimeMs = modelReadyTime - connectionStartTime;
                logServerMessage(`⏱️ LATENCY METRIC - Model Load Time: ${modelLoadTimeMs.toFixed(2)} ms`);
                
                // Store for statistics
                latencyStats.modelLoadTimes.push(modelLoadTimeMs);
                console.log("Model loading times:", latencyStats.modelLoadTimes);
                
                // Update the UI display
                updateLatencyMetricsDisplay();

                // Send start_listening command after connection is established
                const startListeningPayload = {
                    command: "start_listening"
                };
                websocket.send(JSON.stringify(startListeningPayload));
                logServerMessage("📡 Sent 'start_listening' command to server.");
                
            } else if (messageData.event === "listening_started") {
                // Server is ready for audio
                logServerMessage("Server confirmed: Listening started.");
                // UI updates for recording state will be handled by startButton logic
            } else if (messageData.event === "realtime_update" || messageData.event === "realtime_stabilized") {
                // Measure time to first transcription with actual text
                if (messageData.data && messageData.data.text && 
                    messageData.data.text.trim() !== "" && 
                    firstAudioChunkTime && 
                    !firstTranscriptionTime) {
                    
                    firstTranscriptionTime = performance.now();
                    const processingLatencyMs = firstTranscriptionTime - firstAudioChunkTime;
                    logServerMessage(`LATENCY METRIC - Audio Processing Latency: ${processingLatencyMs.toFixed(2)} ms`);
                    
                    // Store for statistics
                    latencyStats.processingLatencies.push(processingLatencyMs);
                    console.log("Processing latencies:", latencyStats.processingLatencies);
                    
                    // Update the UI display
                    updateLatencyMetricsDisplay();
                }
                
                if (messageData.data && messageData.data.text) {
                    // Split text into sentences and join with newlines
                    const text = messageData.data.text;
                    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
                    realtimeTranscript.textContent = sentences.join('\n');
                    
                    // Enhanced logging for realtime transcripts
                    const isStabilized = messageData.event === "realtime_stabilized";
                    const status = isStabilized ? "🔒 STABILIZED" : "⏳ UPDATING";
                    const timestamp = new Date().toLocaleTimeString();
                    
                    logServerMessage(`${status} [${timestamp}]: "${text}"`);
                    
                    if (isStabilized) {
                        logServerMessage(`💾 Realtime transcript saved to Azure Files storage`);
                    }
                }
            } else if (messageData.event === "transcript") {
                if (messageData.data && messageData.data.text) {
                    // Split final transcript into sentences as well
                    const text = messageData.data.text;
                    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
                    setFinalTranscript(sentences.join('\n'));
                    
                    // Log enhanced storage information
                    logServerMessage("🎉 FINAL TRANSCRIPT RECEIVED!");
                    logServerMessage(`📝 Main Model Transcript: "${text}"`);
                    
                    if (messageData.data.language) {
                        logServerMessage(`🌍 Detected Language: ${messageData.data.language} (${messageData.data.language_probability || 'N/A'})`);
                    }
                    
                    if (messageData.data.audio_url) {
                        logServerMessage(`🎵 Audio URL: ${messageData.data.audio_url}`);
                    }
                    
                    if (messageData.data.session_summary) {
                        const summary = messageData.data.session_summary;
                        logServerMessage(`📊 SESSION SUMMARY:`);
                        logServerMessage(`   - Session ID: ${summary.session_id}`);
                        logServerMessage(`   - Total Duration: ${summary.total_duration}s`);
                        logServerMessage(`   - Audio Chunks: ${summary.total_chunks}`);
                        logServerMessage(`   - Realtime Transcripts: ${summary.realtime_transcripts_count}`);
                        logServerMessage(`✅ Azure Files Storage: Session data saved successfully!`);
                    }
                }
            } else if (messageData.event === "error") {
                const errorMsg = messageData.data ? messageData.data.message : 'Unknown error';
                logServerMessage(`Server Error: ${errorMsg}`);
                
                // Special handling for out-of-memory errors
                if (errorMsg.includes("memory") || errorMsg.includes("OOM") || errorMsg.includes("resource")) {
                    logServerMessage("It looks like the server is running out of memory. Try using a smaller model like 'tiny' or 'base'.");
                }
            }
            // Add more event handlers as needed (e.g., for recording_start, voice_activity_start etc.)

        } catch (error) {
            logServerMessage(`Error processing message from server: ${error}`);
            console.error("Error parsing server message:", error, "Raw data:", event.data);
        }
    };

    websocket.onerror = (error) => {
        // Clear connection timeout
        clearTimeout(connectionTimeoutId);
        
        updateStatus("Connection Error");
        logServerMessage(`WebSocket Error: ${error.message || 'An unknown error occurred.'}`);
        console.error("WebSocket Error:", error);
        connectButton.disabled = false;
        startButton.disabled = true;
        stopButton.disabled = true;
    };

    websocket.onclose = (event) => {
        // Clear connection timeout
        clearTimeout(connectionTimeoutId);
        
        updateStatus("Disconnected");
        logServerMessage(`WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason || 'No reason specified'}`);
        
        // Interpret common WebSocket close codes
        if (event.code === 1006) {
            logServerMessage("The connection was closed abnormally. This may be due to server overload, network issues, or insufficient resources for the selected model.");
            logServerMessage("Try using a smaller model like 'tiny' or 'base', or check your network connection.");
        } else if (event.code === 1011) {
            logServerMessage("The server encountered an unexpected condition that prevented it from fulfilling the request. This might be due to model loading failure or memory issues.");
        }
        
        connectButton.disabled = false;
        startButton.disabled = true;
        stopButton.disabled = true;
        websocket = null;
        // Clean up audio resources if they were active
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
        }
        audioContext = null;
        mediaStreamSource = null;
    };
});

startButton.addEventListener('click', async () => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        logServerMessage("WebSocket is not connected. Please connect first.");
        return;
    }

    if (audioContext && audioContext.state === 'running') {
        logServerMessage("Audio recording is already in progress.");
        return;
    }

    // Reset audio processing latency measurement
    firstAudioChunkTime = null;
    firstTranscriptionTime = null;
    isMeasuringFirstChunk = true;

    logServerMessage("Start Recording button clicked. Requesting microphone access...");
    updateStatus("Requesting microphone...");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: TARGET_SAMPLE_RATE
            } 
        });
        
        logServerMessage("Microphone access granted.");
        updateStatus("Microphone active.");

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: TARGET_SAMPLE_RATE
        });
        
        mediaStreamSource = audioContext.createMediaStreamSource(stream);

        try {
            // Instead of using a Blob URL, use a simpler approach with ScriptProcessorNode
            // which has better cross-origin compatibility
            if (!audioContext.createScriptProcessor) {
                throw new Error("ScriptProcessorNode not supported in this browser");
            }
            
            const bufferSize = 2048;
            const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                const inputBuffer = audioProcessingEvent.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                if (isMeasuringFirstChunk) {
                    firstAudioChunkTime = performance.now();
                    isMeasuringFirstChunk = false;
                }
                
                // Convert to 16-bit PCM
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Send the audio data to the WebSocket server
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    // Convert the audio buffer to base64
                    const audioData = new Uint8Array(pcmData.buffer);
                    const base64Audio = btoa(String.fromCharCode.apply(null, audioData));
                    
                    // Create the audio command payload
                    const audioPayload = {
                        command: "send_audio",
                        audio: base64Audio,
                        sample_rate: TARGET_SAMPLE_RATE
                    };
                    websocket.send(JSON.stringify(audioPayload));
                    
                    // Log audio chunk info (less frequent to avoid spam)
                    if (Math.random() < 0.1) { // Log ~10% of chunks
                        logServerMessage(`🎤 Audio chunk sent: ${audioData.length} bytes, ${TARGET_SAMPLE_RATE}Hz`);
                    }
                }
            };

            mediaStreamSource.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
            
            startButton.disabled = true;
            stopButton.disabled = false;
            updateStatus("Recording...");
            logServerMessage("Started recording and processing audio.");
            
        } catch (error) {
            logServerMessage(`Error creating audio processor: ${error.message}`);
            updateStatus("Audio Processing Error");
            console.error("Audio processing error:", error);
        }

    } catch (error) {
        logServerMessage(`Error accessing microphone: ${error.message}`);
        updateStatus("Microphone Error");
        console.error("Microphone access error:", error);
    }
});

stopButton.addEventListener('click', () => {
    if (audioContext) {
        // Send stop_listening command before closing audio context
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const stopListeningPayload = {
                command: "stop_listening"
            };
            websocket.send(JSON.stringify(stopListeningPayload));
            logServerMessage("📡 Sent 'stop_listening' command to server.");

            // Request final transcript
            setTimeout(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    logServerMessage("🔄 Requesting final transcript and session finalization...");
                    logServerMessage("💾 This will trigger Azure Files storage of complete session data");
                    const getTranscriptPayload = {
                        command: "get_transcript"
                    };
                    websocket.send(JSON.stringify(getTranscriptPayload));
                    logServerMessage("📡 Sent 'get_transcript' command to server.");
                }
            }, 1000); // Wait 1 second for processing to complete
        }

        // Disconnect all audio nodes
        if (mediaStreamSource) {
            mediaStreamSource.disconnect();
        }
        
        // Stop all audio tracks from the stream
        if (mediaStreamSource && mediaStreamSource.mediaStream) {
            const tracks = mediaStreamSource.mediaStream.getTracks();
            tracks.forEach(track => track.stop());
        }
        
        audioContext.close().then(() => {
            logServerMessage("Audio context closed successfully.");
        }).catch(error => {
            logServerMessage(`Error closing audio context: ${error.message}`);
        });
        
        audioContext = null;
        mediaStreamSource = null;
    }
    
    startButton.disabled = false;
    stopButton.disabled = true;
    updateStatus("Stopped");
    logServerMessage("Stopped recording.");
});

function updateStatus(message) {
    statusMessage.textContent = message;
}

function logServerMessage(message) {
    const p = document.createElement('p');
    if (typeof message === 'object') message = JSON.stringify(message);
    p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    serverMessages.insertBefore(p, serverMessages.firstChild); // Add to top
    
    // Check if this is a latency metric message and update the display
    if (message.includes('LATENCY METRIC')) {
        updateLatencyMetricsDisplay();
    }
}

function appendRealtimeTranscript(text) { // This might be replaced by direct setting in onmessage
    realtimeTranscript.textContent += text;
}

function setFinalTranscript(text) {
    finalTranscript.textContent = text;
}

// Helper function to downsample audio
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) {
        return buffer;
    }
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0,
            count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

// Helper function to convert Float32 array to 16-bit PCM (Int16Array)
function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
}

// Helper function to format time in both ms and seconds
function formatTimeWithUnits(milliseconds) {
    if (milliseconds === null || milliseconds === undefined) return 'N/A';
    const seconds = milliseconds / 1000;
    return `${milliseconds.toFixed(2)} ms (${seconds.toFixed(3)} s)`;
}

// Helper function to calculate average of an array
function calculateAverage(arr) {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
}

// Update the latency metrics display
function updateLatencyMetricsDisplay() {
    // Update individual metrics
    const modelLoadTimeEl = document.getElementById('modelLoadTime');
    const audioLatencyEl = document.getElementById('audioLatency');
    const avgModelLoadEl = document.getElementById('avgModelLoad');
    const avgAudioLatencyEl = document.getElementById('avgAudioLatency');
    
    // Current values
    if (modelReadyTime && connectionStartTime) {
        const latency = modelReadyTime - connectionStartTime;
        modelLoadTimeEl.textContent = formatTimeWithUnits(latency);
    }
    
    if (firstTranscriptionTime && firstAudioChunkTime) {
        const latency = firstTranscriptionTime - firstAudioChunkTime;
        audioLatencyEl.textContent = formatTimeWithUnits(latency);
    }
    
    // Average values
    const avgModelLoad = calculateAverage(latencyStats.modelLoadTimes);
    if (latencyStats.modelLoadTimes && latencyStats.modelLoadTimes.length > 0) {
        avgModelLoadEl.textContent = `${formatTimeWithUnits(avgModelLoad)} (${latencyStats.modelLoadTimes.length} samples)`;
    } else {
        avgModelLoadEl.textContent = 'N/A';
    }
    
    const avgAudioLatency = calculateAverage(latencyStats.processingLatencies);
    if (latencyStats.processingLatencies && latencyStats.processingLatencies.length > 0) {
        avgAudioLatencyEl.textContent = `${formatTimeWithUnits(avgAudioLatency)} (${latencyStats.processingLatencies.length} samples)`;
    } else {
        avgAudioLatencyEl.textContent = 'N/A';
    }
}

// Set up reset metrics button
const resetMetricsButton = document.getElementById('resetMetricsButton');
if (resetMetricsButton) {
    resetMetricsButton.addEventListener('click', () => {
        // Clear the statistics
        latencyStats.modelLoadTimes = [];
        latencyStats.processingLatencies = [];
        
        // Reset the current measurements
        connectionStartTime = null;
        modelReadyTime = null;
        firstAudioChunkTime = null;
        firstTranscriptionTime = null;
        isMeasuringFirstChunk = true;
        
        // Reset the display
        document.getElementById('modelLoadTime').textContent = 'N/A';
        document.getElementById('audioLatency').textContent = 'N/A';
        document.getElementById('avgModelLoad').textContent = 'N/A';
        document.getElementById('avgAudioLatency').textContent = 'N/A';
        
        logServerMessage("Latency metrics have been reset.");
    });
}

// Initialize button states or other UI aspects if needed
updateStatus("Ready to connect."); 