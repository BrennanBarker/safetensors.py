from pathlib import Path
import tarfile
import json
from tqdm import tqdm

import safetensors.numpy
import numpy as np

archives_dir = 'archives'
out_file = 'np_sd3_medium.safetensors'

archives_dir = Path(archives_dir)
archives = archives_dir.glob('*.tar.gz')
with open(archives_dir / 'metadata.json') as f:
    metadata = json.load(f)
extracted_dirs = Path('extracted')
extracted_dirs.mkdir()
tensors = {}
for i, archive in tqdm(enumerate(archives)):
    extracted_dir = extracted_dirs / f'{i}'
    extracted_dir.mkdir()
    with tarfile.open(archive) as tarfile:
        tarfile.extractall(extracted_dir)
    for npy_file in extracted_dir.glob('*.npy'):
        key = npy_file.stem
        tensors[key] = np.load(npy_file)
safetensors.numpy.save_file(tensors, out_file, metadata=metadata)