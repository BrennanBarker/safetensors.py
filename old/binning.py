from uuid import uuid4
import json

def pack(key_sizes, max_size):
    key_sizes = dict(sorted(key_sizes.items(), reverse=True, key=lambda x: x[1]))
    biggies = []
    for k,v in key_sizes.items():
        if v > max_size:
            print(f'{k} bigger than max size, skipping')
            biggies.append(k)
    with open('biggies.json', 'w') as f:
        json.dump(biggies, f)
    for biggie in biggies:
        key_sizes.pop(biggie)
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

    return list(bins.values())

def load_plan():
    with open('packing_plan_with_sizes.json') as f:
        return json.load(f)
    

if __name__ == '__main__':
    items = {'a':4, 'b': 3, 'c': 2, 'd': 6, 'e': 5, 'f': 1}
    bin_capacity = 6
    bins =pack(items, bin_capacity)
    print(bins)

## STOPPED HERE: TRY NUMPY