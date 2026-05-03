import requests, os

models_to_check = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "nim/meta/llama-3.2-11b-vision-instruct",
    "nim/meta/llama-3.2-90b-vision-instruct",
]

for model in models_to_check:
    resp = requests.get(
        f"https://api.together.xyz/v1/models/{model}",
        headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}
    )
    data = resp.json()
    pricing = data.get("pricing", {})
    print(f"{model}: {pricing}")
