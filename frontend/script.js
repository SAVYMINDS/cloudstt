// WebSocket Server URL
const WS_URL = "wss://cloudstt-api.lemonforest-4eb4d55a.eastus2.azurecontainerapps.io/v1/transcribe";

// DOM Elements
const connectButton = document.getElementById('connectButton');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusMessage = document.getElementById('statusMessage');
const serverMessages = document.getElementById('serverMessages');
const realtimeTranscript = document.getElementById('realtimeTranscript');
const finalTranscript = document.getElementById('finalTranscript');

// WebSocket object
let websocket = null;

// Audio context and processor
let audioContext = null;
let scriptProcessor = null;
let mediaStreamSource = null;
const BUFFER_SIZE = 4096;
const TARGET_SAMPLE_RATE = 16000;

// Event Listeners
connectButton.addEventListener('click', () => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        logServerMessage("Already connected.");
        return;
    }

    updateStatus("Connecting...");
    logServerMessage(`Attempting to connect to ${WS_URL}`);
    connectButton.disabled = true;

    websocket = new WebSocket(WS_URL);

    websocket.onopen = (event) => {
        updateStatus("Connected");
        logServerMessage("Successfully connected to WebSocket server.");
        startButton.disabled = false;
        stopButton.disabled = true; // Should be disabled until recording starts
        // Send the initial 'connect' command to the server
        const connectPayload = {
            command: "connect",
            config: {
                mode: "realtime",
                model: {
                    name: "tiny", // Or make this configurable via UI later
                    language: "en", // Or make this configurable
                    compute_type: "float32"
                },
                realtime_config: {
                    vad: {
                        silero_sensitivity: 0.4
                    }
                }
            }
        };
        websocket.send(JSON.stringify(connectPayload));
        logServerMessage("Sent 'connect' command to server.");
    };

    websocket.onmessage = (event) => {
        try {
            const messageData = JSON.parse(event.data);
            logServerMessage(`Received: ${JSON.stringify(messageData)}`);

            // Handle different event types from server
            if (messageData.event === "connected") {
                // Server confirmed connection and initialization
                logServerMessage(`Session ID: ${messageData.session_id}`);
            } else if (messageData.event === "listening_started") {
                // Server is ready for audio
                logServerMessage("Server confirmed: Listening started.");
                // UI updates for recording state will be handled by startButton logic
            } else if (messageData.event === "realtime_update" || messageData.event === "realtime_stabilized") {
                if (messageData.data && messageData.data.text) {
                    // For simplicity, we'll just replace the content. 
                    // A more sophisticated UI might append or manage segments.
                    realtimeTranscript.textContent = messageData.data.text;
                }
            } else if (messageData.event === "transcript") {
                if (messageData.data && messageData.data.text) {
                    setFinalTranscript(messageData.data.text);
                }
            } else if (messageData.event === "error") {
                logServerMessage(`Server Error: ${messageData.data ? messageData.data.message : 'Unknown error'}`);
            }
            // Add more event handlers as needed (e.g., for recording_start, voice_activity_start etc.)

        } catch (error) {
            logServerMessage(`Error processing message from server: ${error}`);
            console.error("Error parsing server message:", error, "Raw data:", event.data);
        }
    };

    websocket.onerror = (error) => {
        updateStatus("Connection Error");
        logServerMessage(`WebSocket Error: ${error.message || 'An unknown error occurred.'}`);
        console.error("WebSocket Error:", error);
        connectButton.disabled = false;
        startButton.disabled = true;
        stopButton.disabled = true;
    };

    websocket.onclose = (event) => {
        updateStatus("Disconnected");
        logServerMessage(`WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason || 'No reason specified'}`);
        connectButton.disabled = false;
        startButton.disabled = true;
        stopButton.disabled = true;
        websocket = null;
        // Clean up audio resources if they were active
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
        }
        audioContext = null;
        scriptProcessor = null;
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

    logServerMessage("Start Recording button clicked. Requesting microphone access...");
    updateStatus("Requesting microphone...");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        logServerMessage("Microphone access granted.");
        updateStatus("Microphone active.");

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        mediaStreamSource = audioContext.createMediaStreamSource(stream);

        // Create a ScriptProcessorNode for direct audio processing
        // Arguments for createScriptProcessor: bufferSize, numberOfInputChannels, numberOfOutputChannels
        // BUFFER_SIZE is defined above (e.g., 4096)
        scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                return;
            }

            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);

            // 1. Downsample the audio
            const downsampledBuffer = downsampleBuffer(inputData, audioContext.sampleRate, TARGET_SAMPLE_RATE);

            // 2. Convert to 16-bit PCM
            const pcmDataArray = floatTo16BitPCM(downsampledBuffer);
            
            // 3. Base64 encode the PCM data
            // The pcmDataArray is an Int16Array. We need its underlying ArrayBuffer to get bytes.
            // Then convert bytes to base64 string.
            const pcmBytes = new Uint8Array(pcmDataArray.buffer);
            const base64Audio = btoa(String.fromCharCode.apply(null, pcmBytes));

            if (audioProcessingEvent.ครั้ง === undefined) { 
                console.log(`onaudioprocess: Input SR ${audioContext.sampleRate}, Target SR ${TARGET_SAMPLE_RATE}`);
                console.log(`Input samples: ${inputData.length}, Downsampled samples: ${downsampledBuffer.length}, PCM bytes: ${pcmBytes.length}`);
                // Log a snippet of PCM data for debugging
                console.log("PCM Data Snippet (first 10 values):", pcmDataArray.slice(0, 10));
                logServerMessage(`Sending audio chunk (downsampled to ${TARGET_SAMPLE_RATE} Hz)...`);
                audioProcessingEvent.ครั้ง = 0;
            }
            audioProcessingEvent.ครั้ง++;
            if (audioProcessingEvent.ครั้ง > (TARGET_SAMPLE_RATE / BUFFER_SIZE / 2) ) audioProcessingEvent.ครั้ง = undefined; // Log approx twice per second
            
            // 4. Send via WebSocket
            const sendAudioPayload = {
                command: "send_audio",
                audio: base64Audio,
                sample_rate: TARGET_SAMPLE_RATE
            };
            websocket.send(JSON.stringify(sendAudioPayload));
        };

        // Connect the nodes: mediaStreamSource -> scriptProcessor -> audioContext.destination
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination); // Necessary to keep the processor running

        logServerMessage("Audio processing node created and connected. Capturing audio...");

        // Send the 'start_listening' command to the server
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const startListeningPayload = { command: "start_listening" };
            websocket.send(JSON.stringify(startListeningPayload));
            logServerMessage("Sent 'start_listening' command to server.");
        } else {
            logServerMessage("Cannot send 'start_listening', WebSocket not open.");
            //  Should we attempt to stop/cleanup audioContext if WS is not open here?
            // For now, let's assume WS should be open if this button is enabled.
        }

        startButton.disabled = true;
        stopButton.disabled = false;

        // We need to store the stream from getUserMedia to stop its tracks later
        window.localStream = stream;

    } catch (error) {
        console.error("Error accessing microphone:", error);
        logServerMessage(`Error accessing microphone: ${error.message}`);
        updateStatus("Microphone access denied or error.");
    }
});

stopButton.addEventListener('click', () => {
    logServerMessage("Stop Recording button clicked.");
    updateStatus("Stopping recording...");

    // 1. Stop client-side audio capture and processing
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor.onaudioprocess = null; // Remove the handler
        scriptProcessor = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        // Stop all tracks from the media stream used by getUserMedia
        if (window.localStream) { // Assuming you store the stream from getUserMedia globally or pass it
            window.localStream.getTracks().forEach(track => track.stop());
            window.localStream = null; // Clear the stream
        }
        audioContext.close().then(() => {
            logServerMessage("AudioContext closed.");
        });
        audioContext = null;
    }

    // 2. Send 'stop_listening' command to the server
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        const stopListeningPayload = { command: "stop_listening" };
        websocket.send(JSON.stringify(stopListeningPayload));
        logServerMessage("Sent 'stop_listening' command to server.");

        // 3. After a brief moment (or ideally after receiving 'listening_stopped' event),
        // send 'get_transcript' to get the final consolidated text.
        // For simplicity here, we'll use a timeout. A more robust solution would wait for the server's ack.
        setTimeout(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const getTranscriptPayload = { command: "get_transcript" };
                websocket.send(JSON.stringify(getTranscriptPayload));
                logServerMessage("Sent 'get_transcript' command to server.");
            }
        }, 500); // 500ms delay - adjust as needed

    } else {
        logServerMessage("WebSocket not open. Cannot send stop_listening or get_transcript.");
    }

    startButton.disabled = false;
    stopButton.disabled = true;
    updateStatus("Recording stopped. Ready to start new recording.");
    // Optionally clear the realtime transcript area
    // realtimeTranscript.textContent = ""; 
});

function updateStatus(message) {
    statusMessage.textContent = message;
}

function logServerMessage(message) {
    const p = document.createElement('p');
    if (typeof message === 'object') message = JSON.stringify(message);
    p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    serverMessages.insertBefore(p, serverMessages.firstChild); // Add to top
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

// Initialize button states or other UI aspects if needed
updateStatus("Ready to connect."); 