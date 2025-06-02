#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend for testing the enhanced realtime storage.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def serve_frontend():
    """Serve the frontend on localhost:8080"""
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    PORT = 8080
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for local development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"🚀 Starting frontend server...")
            print(f"📁 Serving from: {frontend_dir}")
            print(f"🌐 Frontend URL: http://localhost:{PORT}")
            print(f"🎯 API Endpoint: wss://cloudstt-api-gpu.calmhill-33e39416.westus3.azurecontainerapps.io/v1/transcribe")
            print(f"")
            print(f"📋 Testing Instructions:")
            print(f"   1. Open http://localhost:{PORT} in your browser")
            print(f"   2. Click 'Connect' to establish WebSocket connection")
            print(f"   3. Click 'Start Recording' and speak for 10-30 seconds")
            print(f"   4. Click 'Stop Recording' to get final transcript")
            print(f"   5. Check the logs for Azure Files storage confirmation")
            print(f"")
            print(f"🔍 Monitor server logs with:")
            print(f"   az containerapp logs show --name cloudstt-api-gpu --resource-group mvpv1 --tail 50")
            print(f"")
            print(f"Press Ctrl+C to stop the server")
            print(f"=" * 80)
            
            # Try to open browser automatically
            try:
                webbrowser.open(f"http://localhost:{PORT}")
                print(f"🌐 Browser opened automatically")
            except:
                print(f"⚠️ Could not open browser automatically")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use")
            print(f"💡 Try: lsof -ti:{PORT} | xargs kill")
            print(f"💡 Or open http://localhost:{PORT} directly if server is already running")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    serve_frontend() 