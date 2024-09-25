import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

def numpy_to_parquet(array, key, output_dir: str | Path ='.'):
    output_dir = Path(output_dir)
    filepath = output_dir / f'{key}__{"_".join(map(str, array.shape))}.parquet'
    df = pd.DataFrame(array.flatten())
    pq.write_table(pa.Table.from_pandas(df), filepath, compression='NONE')
    return filepath

def parquet_to_numpy(filepath):
    filepath = Path(filepath)
    table_out = pq.read_table(filepath)
    shape = tuple(map(int, filepath.stem.split('__')[1].split('_')))
    df = table_out.to_pandas()
    flat_array_out = df.to_numpy().flatten()
    return flat_array_out.reshape(shape)

if __name__ == '__main__':
    array = np.array([1,2,3], dtype=np.float16)
    numpy_to_parquet(array, key='test', output_dir='testfile')
    array_out = parquet_to_numpy('testfile/test__3.parquet')
    (array == array_out).all()