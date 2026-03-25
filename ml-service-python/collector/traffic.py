import requests
import time
import random

urls = [
    "https://www.google.com",
    "https://api.github.com",
    "https://example.com",
    "https://www.wikipedia.org",
    "https://jsonplaceholder.typicode.com/posts"
]

session = requests.Session()

while True:
    url = random.choice(urls)
    start = time.time()
    try:
        response = session.get(url, timeout=5)
        latency = time.time() - start
        print(f"[OK] {url} | Status: {response.status_code} | Latency: {latency:.3f}s")
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] {url}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {url} | {e}")
    
    # sleep between 1–3 seconds randomly
    time.sleep(random.uniform(1, 3))