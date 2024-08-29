import huggingface_hub
import os

file_path = huggingface_hub.hf_hub_download('stabilityai/stable-diffusion-3-medium', filename='text_encoders/clip_l.safetensors', token=os.environ['HF_ACCESS_KEY'])
print(file_path)