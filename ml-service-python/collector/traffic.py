import requests
import time
import random
import csv
from datetime import datetime

urls = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost/api",
    "http://localhost/test",
    "https://httpbin.org/get",
    "https://jsonplaceholder.typicode.com/posts",
    "https://example.com"
]

file = "traffic_dataset.csv"

# create csv header
with open(file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "url", "latency", "status", "failure"])

while True:

    url = random.choice(urls)
    start = time.time()

    try:
        response = requests.get(url, timeout=1)

        latency = time.time() - start
        status = response.status_code
        failure = 0

        print(f"{url} | {status} | {latency:.3f}")

    except requests.RequestException:

        latency = None
        status = 0
        failure = 1

        print(f"{url} | Failed")

    timestamp = datetime.now()

    # save to dataset
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, url, latency, status, failure])

    time.sleep(random.uniform(0.5, 2))