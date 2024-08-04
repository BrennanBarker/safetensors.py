from math import prod
import json
from pathlib import Path

import safetensors

from old.binning import pack
from old.metadata import get_header
from old.tar_archive import create_archive


file = 'sd3_medium.safetensors'

print('getting metadata')
header, _ = get_header(file)
metadata = header.pop('__metadata__')
# STOPPED HERE: compression ratio?
sizes = {k:prod(v['shape']) * 5.5 for k,v in header.items()}

print('generating packing plan')
packing_plan = pack(sizes, 200 * 1028 * 1028)
packing_plan_with_sizes = [[(key, sizes[key]) for key in plan] for plan in packing_plan]
with open('packing_plan_with_sizes.json', 'w') as f:
    json.dump(packing_plan_with_sizes, f)

print('loading safetensor file')
tensors = safetensors.safe_open(file, framework='numpy')

print('writing archives')
archives_dir = Path('archives')
archives_dir.mkdir(exist_ok=True)

for i, key_group in enumerate(packing_plan):
    archive_path = archives_dir / f'archive_{i}.tar.gz'
    if not archive_path.exists():
        create_archive(key_group, archive_path, tensors)
    else:
        print(f'{str(archive_path)} found, skipping...')

# smallest = min((prod(v['shape']), k) for k,v in header.items())
# with open('packing_plan_with_sizes.json') as f:
#     pp = json.load(f)

# key_a = 'model.diffusion_model.joint_blocks.0.context_block.mlp.fc1.weight'
# a = get_tensor(file, 'model.diffusion_model.joint_blocks.0.context_block.mlp.fc1.weight')