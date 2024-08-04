import safetensors
from tqdm import tqdm

from old.metadata import get_header, get_header_size

np = 'np_sd3_medium.safetensors'
with open(np, 'rb') as f:
    np_header_size = get_header_size(f)
np_header = get_header(np)
np_metadata = np_header.pop('__metadata__')

tr ='sd3_medium.safetensors'
with open(tr, 'rb') as f:
    tr_header_size = get_header_size(f)
tr_header = get_header(tr)
tr_metadata = tr_header.pop('__metadata__')

assert np_header == tr_header
assert np_metadata == tr_metadata

np_tensors = safetensors.safe_open(np, framework='numpy')
tr_tensors = safetensors.safe_open(tr, framework='numpy')
for key in tqdm(list(tr_header.keys())):
    assert (np_tensors.get_tensor(key) == tr_tensors.get_tensor(key)).all()