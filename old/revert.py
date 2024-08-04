import json
from pathlib import Path

import safetensors.numpy
import numpy as np


def revert(dir: Path | str, out_file: Path | str) -> None:
    dir, out_file = Path(dir), Path(out_file)
    tensors = {}
    files = sorted(list(dir.iterdir()))
    for file in files:
        print(f'working on {file.name}')
        if file.suffix == '.json':
            key = file.stem
            with open(file) as f:
                tensors[key] = np.array(json.load(f), dtype='float32')
        elif file.suffix == '.txt':
            key = file.name.split('_chunk')[0]
            if key not in tensors:
                with open(file) as f:
                    tensors[key] = f.read()
            else:
                with open(file) as f:
                    tensors[key] += f.read()
    for key, value in tensors.items():
        if isinstance(value, str):
            tensors[key] = np.array(json.loads(tensors[key]), dtype='float32')
    safetensors.numpy.save_file(tensors, out_file, metadata={'format': 'pt'})

if __name__ == '__main__':
    dir = 'split_out'
    out_file = 'out.safetensors'
    revert(dir, out_file)
