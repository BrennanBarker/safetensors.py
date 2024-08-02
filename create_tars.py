import os
import tarfile

from tqdm import tqdm

def create_tar_gz_files(input_directory, output_directory, max_size):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)]
    archive_index = 1
    current_size = 0
    tar = None

    for file in tqdm(files):
        file_size = os.path.getsize(file)
        
        if (current_size + file_size) > max_size * 5 or tar is None:  # / 5 accounts for compression
            if tar:
                tar.close()
            archive_name = os.path.join(output_directory, f'archive_{archive_index}.tar.gz')
            tar = tarfile.open(archive_name, 'w:gz')
            archive_index += 1
            current_size = 0
        
        tar.add(file, arcname=os.path.basename(file))
        current_size += file_size

    if tar:
        tar.close()

input_directory = 'out'
output_directory = 'archive_out'
max_size = 200 * 1024 * 1024  # 200 MB in bytes

create_tar_gz_files(input_directory, output_directory, max_size)