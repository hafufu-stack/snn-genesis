"""
Phase 62 Auto-Launcher: Chain Phase 62 after Phase 60-61 batch
================================================================

This script monitors for phase61_log.json to appear (Phase 61 complete),
then kills the batch process and launches Phase 62 (noise half-life experiment).

Usage: Run this in a SEPARATE terminal while phase60_61_batch.py is running.
    python experiments/phase62_launcher.py
"""

import os
import time
import subprocess
import signal

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PHASE61_LOG = os.path.join(RESULTS_DIR, "phase61_log.json")
PHASE62_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase62_noise_halflife.py")

print("=" * 60)
print("  Phase 62 Auto-Launcher")
print("  Watching for Phase 61 completion...")
print(f"  Target: {PHASE61_LOG}")
print("=" * 60)

# Phase 1: Wait for phase61_log.json
check_count = 0
while not os.path.exists(PHASE61_LOG):
    time.sleep(3)  # Check every 3 seconds (must be fast: only ~10s before hibernation!)
    check_count += 1
    if check_count % 20 == 0:  # Print status every ~60 seconds
        print(f"  [{time.strftime('%H:%M:%S')}] Waiting for Phase 61...")

print(f"\n  [{time.strftime('%H:%M:%S')}] Phase 61 COMPLETE! Log detected.")
print("  Stopping batch process...")

# Find python processes running phase60_61_batch.py
try:
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | "
         "Where-Object {$_.CommandLine -like '*phase60_61*'} | "
         "Stop-Process -Force"],
        capture_output=True, text=True, timeout=10
    )
    print(f"  Kill result: {result.stdout.strip() if result.stdout else 'Process killed'}")
except Exception as e:
    print(f"  Kill attempt: {e}")

# Also try to cancel any pending task
time.sleep(3)

# Phase 3: Launch Phase 62
print(f"\n  [{time.strftime('%H:%M:%S')}] Launching Phase 62: Noise Half-Life!")
print("=" * 60)

# Use os.system so it blocks until Phase 62 completes
os.system(f"python \"{PHASE62_SCRIPT}\"")

print(f"\n  [{time.strftime('%H:%M:%S')}] Phase 62 launcher complete.")
