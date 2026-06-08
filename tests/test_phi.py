import requests

def call_phi(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi4-mini",
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


print(
    call_phi(
        'Return only JSON: {"status":"working"}'
    )
)