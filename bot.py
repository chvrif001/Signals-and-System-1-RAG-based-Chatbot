import requests, os

resp = requests.get(
    "https://api.together.xyz/v1/models",
    headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}
)

models = resp.json()
vision = [m["id"] for m in models if any(k in m["id"].lower() for k in ["vision", "llama-3.2", "vl", "visual"])]
for m in vision:
    print(m)
