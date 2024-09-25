from math import prod
import tarfile
from pathlib import Path
import json

import safetensors
import numpy as np
from tqdm import tqdm

from code.header import get_header
from pack import pack
from parq import numpy_to_parquet

def create_archive(keys, archive_name, tensors):
    temp = Path('temp')
    temp.mkdir(exist_ok=True)
    with tarfile.open(archive_name, 'w:gz') as tar:
        for key in tqdm(keys):
            array = tensors.get_tensor(key)  # type: ignore 
            filepath = numpy_to_parquet(array, key, temp)
            tar.add(filepath, arcname=filepath.name)
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