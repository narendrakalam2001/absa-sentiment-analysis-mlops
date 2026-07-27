"""
run_dashboard.py — Start the ABSA Streamlit monitoring dashboard.
Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8501
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Monitoring Dashboard")
    parser.add_argument("--port", default=8501, type=int, help="Streamlit port")
    args = parser.parse_args()

    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "monitoring", "monitoring_dashboard.py"
    )

    print(f"\n📊 Starting ABSA Monitoring Dashboard on port {args.port}")
    print(f"   URL: http://localhost:{args.port}\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0",
    ])
