from __future__ import annotations

import requests

payload = {
    "text": (
        "Create an authorized defensive enterprise scenario focused on "
        "phishing awareness and credential protection."
    ),
    "language": "en",
    "max_new_tokens": 384,
    "do_sample": False,
}

response = requests.post(
    "http://localhost:8000/v1/scenarios/generate",
    json=payload,
    timeout=180,
)
response.raise_for_status()
print(response.json())
