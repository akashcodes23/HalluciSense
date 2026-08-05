"""HalluciSense Python SDK Integration Client Example."""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def verify_response(text: str) -> dict:
    """Send text verification request to HalluciSense API."""
    url = f"{BASE_URL}/verification/verify-text"
    payload = {
        "text": text
    }
    headers = {"Content-Type": "application/json"}

    res = requests.post(url, json=payload, headers=headers, timeout=10)
    res.raise_for_status()
    return res.json()


if __name__ == "__main__":
    text = "Paris is the capital and most populous city of France."
    print(f"Submitting query to HalluciSense API ({BASE_URL}/verification/verify-text)...")
    result = verify_response(text)
    print(json.dumps(result, indent=2))
