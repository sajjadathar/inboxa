"""Sends every sample email to /classify and prints a score.

Run it with:
    docker compose exec backend python -m app.samples.test_classify
"""
import json
import time
from pathlib import Path

import httpx

API = "http://localhost:8000/classify"
DELAY = 4  # seconds between calls, to stay under the free-tier rate limit

emails = json.loads((Path(__file__).parent / "emails.json").read_text())
correct = 0

print(f"{'EXPECTED':<9} {'GOT':<9} {'CONF':<5} {'URG':<7} {'LANG':<4} SUBJECT")
print("-" * 80)

for e in emails:
    payload = {"sender": e["sender"], "subject": e["subject"], "body": e["body"]}
    r = httpx.post(API, json=payload, timeout=120)
    if r.status_code != 200:
        print(f"ERROR {r.status_code}: {r.text[:150]}")
        time.sleep(DELAY)
        continue

    c = r.json()
    ok = c["category"] == e["expected"]
    correct += ok
    mark = "  " if ok else "✗ "
    print(
        f"{mark}{e['expected']:<7} {c['category']:<9} {c['confidence']:<5.2f} "
        f"{c['urgency']:<7} {c['language']:<4} {e['subject'][:40]}"
    )
    time.sleep(DELAY)

print("-" * 80)
print(f"Score: {correct}/{len(emails)} correct")
