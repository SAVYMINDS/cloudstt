#import sounddevice as sd
#import base64

#print("Available audio devices:\n")
#try:
#   print(sd.query_devices())
#except Exception as e:
 #   print(f"Could not query devices. Is sounddevice installed and working? Error: {e}")
 #  print("You might need to install it: pip install sounddevice")
  #  print("On macOS, you might also need to grant terminal/Python access to the microphone in System Settings > Privacy & Security > Microphone.")

#print("\n--- How to Interpret ---")
#print("Look for your microphone in the list. The number at the beginning of its line is its index.")
#print("Input devices will have a 'max_input_channels' greater than 0.")
#print("Output devices will have a 'max_output_channels' greater than 0.")
#print("Your default input device often has a '>' or '*' next to its index or name.")
import base64
import json # Added for creating the JSON payload

filename = "/Users/test/Downloads/cloudstt/storage/small.wav"
output_payload_filename = "send_audio_payload.json" # File to save the JSON payload

audio_b64 = ""
try:
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    # print(f"---BASE64 FOR {filename}---")
    # print(audio_b64)
    # print("---END OF BASE64---")
    # print(f"\n(String length: {len(audio_b64)} characters)")

    # Construct the full JSON payload
    send_audio_command = {
        "command": "send_audio",
        "audio": audio_b64,
        "sample_rate": 16000 # Assuming 16kHz for small.wav
    }

    # Save the JSON payload to a file
    with open(output_payload_filename, "w") as outfile:
        json.dump(send_audio_command, outfile, indent=4) # indent for readability
    
    print(f"--- Full JSON payload for send_audio command saved to: {output_payload_filename} ---")
    # Also print to console for convenience
    # print(json.dumps(send_audio_command, indent=4))
    print("You can now copy the content of this file and paste it into wscat.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Please ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")