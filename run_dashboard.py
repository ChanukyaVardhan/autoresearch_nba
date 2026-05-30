#!/usr/bin/env python3
"""Launch the live autoresearch dashboard. Run from .autoresearch_nba/:
    python3 run_dashboard.py [port]
Open the printed URL and leave it open while the loop runs — it refreshes every 2s.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo root, NOT src/
from src.dashboard import start_server

port = int(sys.argv[1]) if len(sys.argv) > 1 else 6060
url = start_server(port)
print(f"LIVE DASHBOARD: {url}  (Ctrl-C to stop)")
while True:
    time.sleep(3600)
