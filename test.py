from safetensors import safe_open
with safe_open("/home/weight/Qwen2.5-VL-3B-Instruct/model-00001-of-00002.safetensors", framework="pt") as f:
    print(f.keys())
