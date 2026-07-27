"""
run_simulation.py — Run the ABSA review simulator.
Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --scenario negative_spike --n 30
    python scripts/run_simulation.py --scenario positive_wave --n 15 --delay 1.0
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from simulation.review_simulator import simulate_reviews

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Review Simulator")
    parser.add_argument(
        "--scenario", default="realistic_mix",
        choices=["realistic_mix", "negative_spike", "positive_wave"],
        help="Simulation scenario"
    )
    parser.add_argument("--n",     default=20,  type=int,   help="Number of reviews to send")
    parser.add_argument("--delay", default=0.5, type=float, help="Delay between requests (s)")
    args = parser.parse_args()

    simulate_reviews(n=args.n, scenario=args.scenario, delay=args.delay)
