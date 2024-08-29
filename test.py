import safetensors
from tqdm import tqdm

from archive import get_header

def test(regenerated_filename, original_filename):
    regenerated_header = get_header(regenerated_filename)
    regenerated_metadata = regenerated_header.pop('__metadata__')
    original_header = get_header(original_filename)
    original_metadata = original_header.pop('__metadata__')

    assert regenerated_header == original_header
    assert regenerated_metadata == original_metadata

    regenerated_tensors = safetensors.safe_open(regenerated_filename, framework='numpy')
    original_tensors = safetensors.safe_open(original_filename, framework='numpy')
    for key in tqdm(list(original_header.keys())):
        assert (regenerated_tensors.get_tensor(key) == original_tensors.get_tensor(key)).all()  # type: ignore

if __name__ == '__main__':
    import sys
    regenerated_filename = sys.argv[1]
    original_filename = sys.argv[2]
    test(regenerated_filename, original_filename)