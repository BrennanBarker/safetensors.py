from uuid import uuid4

def pack(key_sizes, max_size):
    key_sizes = dict(sorted(key_sizes.items(), reverse=True, key=lambda x: x[1]))
    bins = {}

    for key, size in key_sizes.items():
        # Try to place item in an existing bin
        placed = False
        for bin_id, bin_keys in bins.items():
            if sum(key_sizes[key] for key in bin_keys) + size <= max_size:
                bins[bin_id].append(key)
                placed = True
                break
        # If item doesn't fit in any existing bin, create a new bin
        if not placed:
            bins[uuid4().hex] = [key]

    return bins

items = {'a':4, 'b': 3, 'c': 2, 'd': 6, 'e': 5, 'f': 1}
bin_capacity = 6
bins =pack(items, bin_capacity)
print(bins)