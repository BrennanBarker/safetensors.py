import os
import boto3
from pathlib import Path
from tqdm import tqdm

session = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='us-east-1'
)
s3 = session.client('s3')
bucket = 'buck-9ed844ce-f99e-4251-b43c-ad8f36d59a18'

def get_keys():
    resp = s3.list_objects(Bucket=bucket)
    return [obj['Key'] for obj in resp.get('Contents') or []]

def upload_files(dir):
    for file in tqdm(list(Path(dir).iterdir())):
        s3.upload_file(file, bucket, file.name)
