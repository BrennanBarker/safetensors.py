import json
from functools import cache

@cache
def get_header_size(safetensors_file_handle):
    return int.from_bytes(safetensors_file_handle.read(8), byteorder='little')

@cache
def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = get_header_size(f)
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    return header

if __name__ == '__main__':
    file = 'sd3_medium.safetensors'
    header, _ = get_header(file)