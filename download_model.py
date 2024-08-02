import huggingface_hub
import os

file_path = huggingface_hub.hf_hub_download('stabilityai/stable-diffusion-3-medium', filename='sd3_medium.safetensors', token=os.environ['HF_ACCESS_KEY'])
print(file_path)