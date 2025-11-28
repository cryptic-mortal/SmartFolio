import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importing from tools_mcp triggers the __init__.py which registers all the tools
from tools_mcp import SmartFolioMCPServer

if __name__ == "__main__":
    # Use SSE transport to allow external connections (e.g. from interactive_client.py)
    # This allows running the server in one terminal and the client in another.
    server = SmartFolioMCPServer(app_id="smartfolio-xai", transport="sse", port=9123)
    
    print("Server configured for SSE on port 9123")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        sys.stderr.write("Server stopped by user.\n")
    except Exception as e:
        sys.stderr.write(f"Server error: {e}\n")
        sys.exit(1)
