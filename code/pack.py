from uuid import uuid4

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