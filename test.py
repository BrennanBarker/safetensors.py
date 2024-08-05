import safetensors
from tqdm import tqdm

from archive import get_header

re = 'regenerated.safetensors'
re_header = get_header(re)
re_metadata = re_header.pop('__metadata__')

tr ='sd3_medium.safetensors'
tr_header = get_header(tr)
tr_metadata = tr_header.pop('__metadata__')

assert re_header == tr_header
assert re_metadata == tr_metadata

re_tensors = safetensors.safe_open(re, framework='numpy')
tr_tensors = safetensors.safe_open(tr, framework='numpy')
for key in tqdm(list(tr_header.keys())):
    assert (re_tensors.get_tensor(key) == tr_tensors.get_tensor(key)).all()  # type: ignore