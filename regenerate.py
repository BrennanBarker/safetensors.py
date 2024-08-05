from pathlib import Path
import tarfile
import json
from tqdm import tqdm

import safetensors.numpy
import numpy as np

from parq import parquet_to_numpy

def regenerate_safetensors(archives_dir: Path, output_filename: str):
    archives = archives_dir.glob('*.tar.gz')
    with open(archives_dir / 'metadata.json') as f:
        metadata = json.load(f)
    extracted_dirs = Path('extracted')
    extracted_dirs.mkdir()
    tensors = {}
    for i, archive in tqdm(list(enumerate(archives))):
        extracted_dir = extracted_dirs / f'{i}'
        extracted_dir.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(extracted_dir)
        for parquet_file in extracted_dir.glob('*.parquet'):
            key = parquet_file.stem.split('__')[0]
            tensors[key] = parquet_to_numpy(parquet_file)
        for file in extracted_dir.iterdir():
            file.unlink()
        extracted_dir.rmdir()
    extracted_dirs.rmdir()
    safetensors.numpy.save_file(tensors, output_filename, metadata=metadata)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Recreate safetensors file from archives.")
    parser.add_argument('archives_dir', type=Path, help='The path to the archives directory.')
    parser.add_argument('output_filename', type=Path, help='The filename of the regenerated safetensors file.')
    args = parser.parse_args()

    regenerate_safetensors(args.archives_dir, args.output_filename)