import safetensors
import numpy as np
import json

m = safetensors.safe_open('model.safetensors', framework='numpy')
o = safetensors.safe_open('out.safetensors', framework='numpy')

assert m.metadata() == o.metadata()
assert m.keys() == o.keys()
assert all(np.all(m.get_tensor(k) == o.get_tensor(k)) for k in m.keys())
for k in dir(m):
    print(k, getattr(m, k) == getattr(o, k))
o.__sizeof__()
m.__sizeof__()
help(o.__format__)

def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = int.from_bytes(f.read(8), byteorder='little')
        header_bytes = f.read(header_size)
        rest = f.read()
    return json.loads(header_bytes.decode('utf-8')), header_size, header_bytes, rest

m_header, m_header_size, m_header_bytes, m_rest = get_header('model.safetensors')
o_header, o_header_size, o_header_bytes, o_rest = get_header('out.safetensors')


def byte_by_byte_diff(file1_path, file2_path):
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        byte_pos = 0
        while True:
            byte1 = file1.read(1)
            byte2 = file2.read(1)

            if not byte1 and not byte2:
                break  # End of both files

            if byte1 != byte2:
                print(f"Difference at byte {byte_pos}: {byte1} != {byte2}")

            byte_pos += 1

        # Check if one file is longer than the other
        remaining1 = file1.read()
        remaining2 = file2.read()

        if remaining1 or remaining2:
            print(f"Files differ in length. Remaining bytes in file1: {len(remaining1)}, file2: {len(remaining2)}")

# Example usage
file1_path = 'model.safetensors'
file2_path = 'out.safetensors'
byte_by_byte_diff(file1_path, file2_path)

with open('model.safetensors', 'rb') as f: 
    mb_header = f.read(12518)
    mb_rest = f.read()
with open('out.safetensors', 'rb') as f: 
    ob_header = f.read(12518)
    ob_rest = f.read()

with open('out2.safetensors', 'wb') as f:
    f.write(int.to_bytes(m_header_size, length=8, byteorder='little'))
    f.write(m_header_bytes)
    f.write(o_rest)

st2 = safetensors.safe_open('out2.safetensors', framework='numpy')