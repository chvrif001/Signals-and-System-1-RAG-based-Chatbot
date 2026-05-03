import requests
import base64
import os
from PIL import Image
import io

API_KEY = os.environ["TOGETHER_API_KEY"]

# Create a tiny test image (no file needed)
img = Image.new("RGB", (100, 100), color=(255, 0, 0))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
image_bytes = buf.getvalue()

b64 = base64.b64encode(image_bytes).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64}"

payload = {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What colour is this image?"}
            ]
        }
    ]
}

resp = requests.post(
    "https://api.together.xyz/v1/chat/completions",
    json=payload,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    timeout=30
)

print("Status:", resp.status_code)
print("Response:", resp.text)
