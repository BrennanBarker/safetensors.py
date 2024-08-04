from pathlib import Path
import tarfile
import json
from tqdm import tqdm

import safetensors.numpy
import numpy as np

def regenerate_safetensors(archives_dir: Path, output_filename: str):
    archives = archives_dir.glob('*.tar.gz')
    with open(archives_dir / 'metadata.json') as f:
        metadata = json.load(f)
    extracted_dirs = Path('extracted')
    extracted_dirs.mkdir()
    tensors = {}
    for i, archive in tqdm(enumerate(list(archives))):
        extracted_dir = extracted_dirs / f'{i}'
        extracted_dir.mkdir()
        with tarfile.open(archive) as tarfile:
            tarfile.extractall(extracted_dir)
        for npy_file in extracted_dir.glob('*.npy'):
            key = npy_file.stem
            tensors[key] = np.load(npy_file)
    safetensors.numpy.save_file(tensors, output_filename, metadata=metadata)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Recreate safetensors file from archives.")
    parser.add_argument('archives_dir', type=Path, help='The path to the archives directory.')
    parser.add_argument('output_filename', type=Path, help='The filename of the regenerated safetensors file.')
    args = parser.parse_args()

    regenerate_safetensors(args.archives_dir, args.output_filename)