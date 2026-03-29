import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import requests
import urllib3

# List of URLs to test
urls = [
    "https://www.google.com",
    "https://api.github.com",
    "http://example.com",
    "https://www.wikipedia.org",
    "https://jsonplaceholder.typicode.com/posts"
]

dataset_path = Path(__file__).resolve().with_name("traffic_dataset.csv")
verify_ssl = os.getenv("TRAFFIC_VERIFY_SSL", "0") == "1"
max_iterations = int(os.getenv("TRAFFIC_MAX_ITERATIONS", "0"))
sleep_min = float(os.getenv("TRAFFIC_SLEEP_MIN", "1"))
sleep_max = float(os.getenv("TRAFFIC_SLEEP_MAX", "3"))
request_timeout = (
    float(os.getenv("TRAFFIC_CONNECT_TIMEOUT", "3")),
    float(os.getenv("TRAFFIC_READ_TIMEOUT", "5")),
)
headers = {
    "User-Agent": "traffic-collector/1.0",
    "Connection": "close",
}

if not verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create file with header only if it doesn't exist
if not dataset_path.exists():
    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "url",
            "latency",
            "status",
            "failure"
        ])

print(
    f"Traffic collection started. Writing to {dataset_path}. "
    f"SSL verify={'on' if verify_ssl else 'off'}",
    flush=True,
)

try:
    iteration = 0

    while max_iterations <= 0 or iteration < max_iterations:
        iteration += 1
        url = random.choice(urls)
        start = time.perf_counter()
        latency = 0.0
        status = 0
        failure = 1

        try:
            response = requests.get(
                url,
                timeout=request_timeout,
                headers=headers,
                verify=verify_ssl,
            )
            latency = time.perf_counter() - start
            status = response.status_code
            failure = 0
            print(
                f"[OK] #{iteration} {url} | Status: {status} | Latency: {latency:.3f}s",
                flush=True,
            )

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] #{iteration} {url}", flush=True)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] #{iteration} {url} | {e}", flush=True)

        except Exception as e:
            print(f"[UNEXPECTED] #{iteration} {url} | {type(e).__name__}: {e}", flush=True)

        # Append new row to dataset
        try:
            with dataset_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    url,
                    latency,
                    status,
                    failure
                ])
                f.flush()
        except OSError as e:
            print(f"[WRITE ERROR] Could not append to {dataset_path} | {e}", flush=True)

        time.sleep(random.uniform(sleep_min, sleep_max))
except KeyboardInterrupt:
    print("Traffic collection stopped.", flush=True)
