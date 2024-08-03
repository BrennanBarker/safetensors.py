from math import prod
from uuid import uuid4
import tarfile
from tempfile import NamedTemporaryFile
from pathlib import Path

import safetensors
import numpy as np
from tqdm import tqdm

from metadata import get_header

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

filename = 'sd3_medium.safetensors'

header = get_header(filename)
metadata = header.pop('__metadata__')

sizes = { k: prod(v['shape']) * 2 for k,v in header.items() }

packing_plan = pack(sizes, 200 * 1024 ** 2)

tensors = safetensors.safe_open(filename, framework='numpy')

archives = Path('archives')
for i, plan in enumerate(packing_plan):
    create_archive(plan, archives / f'archive{i}.tar.gz', tensors)
