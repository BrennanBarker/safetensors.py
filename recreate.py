import json
import safetensors

safetensors_filename = '/home/codespace/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium/blobs/92db4295e9c9ab8401ef60566d975656a35b0bd0f6d9ce0d083725171f7b3174'

def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = int.from_bytes(f.read(8), byteorder='little')
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    return header

header = get_header(safetensors_filename)
header.pop('__metadata__')
header

tensors = safetensors.safe_open(safetensors_filename, framework='numpy')

from diffusers.loaders.single_file_utils import fetch_diffusers_config, is_t5_in_single_file, convert_sd3_t5_checkpoint_to_diffusers
fetch_diffusers_config(safetensors_filename)

from diffusers.models.model_loading_utils import load_state_dict

sd = load_state_dict(safetensors_filename)

sorted({'.'.join(k.split('.')[0:5]) for k in header.keys() if 'embedder' in k})