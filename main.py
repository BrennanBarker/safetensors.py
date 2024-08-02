from math import prod

from binning import pack
from metadata import get_header


file = 'sd3_medium.safetensors'
header, _ = get_header(file)
metadata = header.pop('__metadata__')
# STOPPED HERE: compression ratio?
sizes = {k:prod(v['shape']) for k,v in header.items()}