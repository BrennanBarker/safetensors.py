import json

def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = int.from_bytes(f.read(8), byteorder='little')
        header_bytes = f.read(header_size)
    return json.loads(header_bytes.decode('utf-8'))



if __name__ == '__main__':
    file = 'sd3_medium.safetensors'
    header = get_header(file)
    