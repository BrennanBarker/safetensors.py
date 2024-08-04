from math import prod
from uuid import uuid4
import tarfile
from pathlib import Path
import json

import safetensors
import numpy as np
from tqdm import tqdm


def get_header(safetensors_file):
    with open(safetensors_file, 'rb') as f: 
        header_size = int.from_bytes(f.read(8), byteorder='little')
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    return header

def pack(key_sizes, max_size):
    key_sizes = dict(sorted(key_sizes.items(), reverse=True, key=lambda x: x[1]))
    bins = {}
    for key, size in key_sizes.items():
        placed = False
        for bin_id, bin_keys in bins.items():
            if sum(key_sizes[key] for key in bin_keys) + size <= max_size:
                bins[bin_id].append(key)
                placed = True
                break
        if not placed:
            bins[uuid4().hex] = [key]
    return list(bins.values())

def create_archive(keys, archive_name, tensors):
    temp = Path('temp')
    temp.mkdir(exist_ok=True)
    with tarfile.open(archive_name, 'w:gz') as tar:
        for key in tqdm(keys):
            filepath = Path(f'temp/{key}.npy')
            array = tensors.get_tensor(key)  # type: ignore 
            np.save(filepath, array, allow_pickle=False)
            tar.add(filepath, arcname=f'{key}.npy')
            filepath.unlink()

def create_archives(safetensors_filename: Path, output_dir: Path) -> None:
    output_dir.mkdir()
    header = get_header(safetensors_filename)
    metadata = header.pop('__metadata__')
    sizes = { k: prod(v['shape']) * 2 for k,v in header.items() }
    packing_plan = pack(sizes, 200 * 1024 ** 2)

    tensors = safetensors.safe_open(safetensors_filename, framework='numpy')

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    for i, plan in enumerate(packing_plan):
        create_archive(plan, output_dir / f'{i}.tar.gz', tensors)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process a safetensors file and save the output to a directory.")
    parser.add_argument('safetensors_filename', type=Path, help='The path to the safetensors file.')
    parser.add_argument('output_dir', type=Path, help='The directory where the output will be saved.')
    args = parser.parse_args()

    create_archives(args.safetensors_filename, args.output_dir)