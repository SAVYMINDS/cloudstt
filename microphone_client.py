import asyncio
import websockets
import json
import base64
import pyaudio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio settings
CHUNK = 4096  # Increased chunk size for potentially better network efficiency
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# WebSocket server URI (using secure Ingress with nip.io domain)
WEBSOCKET_URI = "wss://20-245-245-243.nip.io/v1/transcribe"

async def stream_microphone():
    """Captures audio from the microphone and streams it to the WebSocket server."""

    p = pyaudio.PyAudio()
    stream = None
    websocket = None # Define websocket in the outer scope
    stop_event = asyncio.Event() # Event to signal shutdown

    try:
        logger.info("Opening microphone stream...")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        logger.info("Microphone stream opened successfully.")
    except OSError as e:
        logger.error(f"Error opening microphone stream: {e}")
        logger.error("Please ensure a microphone is connected and configured.")
        logger.error("Available audio devices:")
        try:
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                logger.error(f"  {i}: {dev['name']} (Input Channels: {dev['maxInputChannels']})")
        except Exception as audio_err:
            logger.error(f"Could not list audio devices: {audio_err}")
        p.terminate()
        sys.exit(1) # Exit if microphone cannot be opened

    logger.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URI}")

    try:
        # Assign to the outer scope websocket
        websocket = await websockets.connect(WEBSOCKET_URI, ping_interval=None, ssl=True)
        logger.info("WebSocket connection established.")

        # 1. Send connection command
        connect_msg = {
            "command": "connect",
            "config": {
                "mode": "realtime",
                "model": {"name": "tiny", "language": "en", "compute_type": "float32"},
                "realtime_config": {"vad": {"silero_sensitivity": 0.4}}
            }
        }
        await websocket.send(json.dumps(connect_msg))
        response = await websocket.recv()
        logger.info(f"Connect Response: {response}")
        connect_data = json.loads(response)
        if connect_data.get("event") != "connected":
            logger.error("Failed to get 'connected' event from server.")
            return

        # 2. Send start_listening command
        await websocket.send(json.dumps({"command": "start_listening"}))
        response = await websocket.recv()
        logger.info(f"Start Listening Response: {response}")
        start_data = json.loads(response)
        if start_data.get("event") != "listening_started":
            logger.error("Failed to get 'listening_started' event.")
            return

        logger.info(">>> Microphone streaming started. Speak into the microphone. Press Ctrl+C to stop. <<<")

        # 3. Start streaming loop (send audio, receive results)
        async def send_audio(ws, stop_signal):
            while not stop_signal.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_base64 = base64.b64encode(data).decode('utf-8')
                    audio_msg = {
                        "command": "send_audio",
                        "audio": audio_base64,
                        "sample_rate": RATE
                    }
                    await ws.send(json.dumps(audio_msg))
                    # Introduce a small delay to prevent busy-waiting and allow receiving
                    await asyncio.sleep(0.01)
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Send audio task: WebSocket connection closed.")
                    stop_signal.set() # Signal other tasks to stop
                    break
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    stop_signal.set() # Signal other tasks to stop
                    break
            logger.info("Send audio task finished.")

        async def receive_results(ws, stop_signal):
            while not stop_signal.is_set():
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1) # Short timeout
                    logger.info(f"Received: {response}")
                except asyncio.TimeoutError:
                    continue # No message received, continue checking
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Receive results task: WebSocket connection closed.")
                    stop_signal.set() # Signal other tasks to stop
                    break
                except Exception as e:
                    logger.error(f"Error receiving results: {e}")
                    stop_signal.set() # Signal other tasks to stop
                    break
            logger.info("Receive results task finished.")

        # Run send and receive tasks concurrently
        send_task = asyncio.create_task(send_audio(websocket, stop_event))
        receive_task = asyncio.create_task(receive_results(websocket, stop_event))

        # Wait for tasks to complete OR for KeyboardInterrupt
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        # If tasks finished due to error/closure, log it
        if not send_task.done() or not receive_task.done():
             logger.info("Waiting for tasks to finish after stop signal...")
             await asyncio.wait([send_task, receive_task], timeout=2.0)

        logger.info("Streaming loops finished.")

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running and port-forwarding active ({WEBSOCKET_URI})?")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, initiating graceful shutdown...")
        stop_event.set() # Signal tasks to stop
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        stop_event.set() # Ensure stop on other errors

    finally:
        # 4. Graceful Shutdown and Cleanup
        logger.info("Initiating cleanup sequence...")

        # Ensure tasks are cancelled if they haven't finished
        if 'send_task' in locals() and not send_task.done(): send_task.cancel()
        if 'receive_task' in locals() and not receive_task.done(): receive_task.cancel()
        # Wait briefly for cancellations to be processed
        await asyncio.sleep(0.1)

        # Send stop and get transcript commands if websocket is connected
        if websocket and websocket.open:
            try:
                logger.info("Sending stop_listening command...")
                await websocket.send(json.dumps({"command": "stop_listening"}))
                # Try to receive the stop confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                logger.info(f"Stop Listening Response: {response}")

                logger.info("Sending get_transcript command...")
                await websocket.send(json.dumps({"command": "get_transcript"}))
                # Wait for and print the final transcript
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0) # Longer timeout for final transcript
                logger.info(f"FINAL TRANSCRIPT Response: {response}")

            except asyncio.TimeoutError:
                 logger.warning("Timeout waiting for response during cleanup.")
            except websockets.exceptions.ConnectionClosed:
                 logger.warning("Connection closed during cleanup command sending.")
            except Exception as e:
                 logger.error(f"Error during final command sequence: {e}")
            finally:
                 logger.info("Closing WebSocket connection.")
                 await websocket.close()
        else:
            logger.info("WebSocket connection already closed, skipping final commands.")

        # Close PyAudio stream
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
            logger.info("Microphone stream stopped and closed.")
        if p:
            p.terminate()
            logger.info("PyAudio terminated.")

        logger.info("Client shutdown complete.")

if __name__ == "__main__":
    # Removed the outer try/except KeyboardInterrupt here, handled within stream_microphone
    asyncio.run(stream_microphone()) 