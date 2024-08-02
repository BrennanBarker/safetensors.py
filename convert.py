import json
from pathlib import Path
from s3 import get_keys

import safetensors.numpy

def safetensors_to_json(input_file: Path | str, output_dir: Path | str) -> None:
    input_file, output_dir = Path(input_file), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: use new metadata get_tensor instead of loading whole file
    tensors = safetensors.numpy.load_file(input_file)
    keys_in_s3 = [x.replace('.json', '') for x in get_keys()]

    for key, tensor in tensors.items():
        print(key)
        if key in keys_in_s3:
            print('found in s3.')
            continue
        with open(output_dir / f'{key}.json', 'w') as f:
            json.dump(tensor.tolist(), f)

# def split_file_by_size(input_file, output_dir, size_in_mb):
#     input_file, output_dir = Path(input_file), Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     size_in_bytes = size_in_mb * 1024 * 1024
#     with open(input_file, 'rb') as f:
#         file_count = 0
#         while True:
#             chunk = f.read(size_in_bytes)
#             if not chunk:
#                 break
#             if file_count == 0:
#                 suffix = '.json'
#             else:
#                 suffix = f'_chunk{str(file_count).zfill(3)}.txt'
#                 if file_count == 1:
#                     (output_dir / f'{input_file.stem}.json').rename(output_dir / f'{input_file.stem}_chunk000.txt' )
#             with open(output_dir / f'{input_file.stem}{suffix}', 'wb') as out_file:
#                 out_file.write(chunk)
#             file_count += 1

if __name__ == '__main__':
    input_file = 'sd3_medium.safetensors'
    unsplit_output_dir = Path('out')
    # split_output_dir = Path('split_out')

    safetensors_to_json(input_file, unsplit_output_dir )
    # for file in unsplit_output_dir.glob('*'):
    #     split_file_by_size(file, split_output_dir, 40)

