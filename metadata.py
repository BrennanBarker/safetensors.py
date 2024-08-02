import json
import numpy as np
from functools import cache

@cache
def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = int.from_bytes(f.read(8), byteorder='little')
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    return header, header_size

def get_tensor(safetensors_file, key):
    header, header_size = get_header(safetensors_file)
    metadata = header[key]
    start, end = metadata['data_offsets']
    with open(file, 'rb') as f:
        f.seek(start + header_size + 8)
        return np.frombuffer(f.read(end - start), dtype=np.float16)


if __name__ == '__main__':
    file = 'sd3_medium.safetensors'
    header, _ = get_header(file)
    key = list(header.keys())[-1]
    true = np.load('true.npy')
    tens = get_tensor(file, key)
    assert all(true==tens)