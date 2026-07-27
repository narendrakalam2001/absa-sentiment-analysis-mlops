"""
run_api.py — Start the ABSA FastAPI prediction server.
Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA FastAPI Server")
    parser.add_argument("--host",   default="0.0.0.0",        help="Host address")
    parser.add_argument("--port",   default=8000, type=int,   help="Port number")
    parser.add_argument("--reload", action="store_true",       help="Enable hot reload (dev mode)")
    args = parser.parse_args()

    print(f"\n🚀 Starting ABSA API on http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   Health: http://{args.host}:{args.port}/health\n")

    uvicorn.run(
        "serving.absa_api:app",
        host   = args.host,
        port   = args.port,
        reload = args.reload,
    )
