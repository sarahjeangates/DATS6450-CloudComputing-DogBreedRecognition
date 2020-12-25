import boto3
import os


def upload_files(path):
    session = boto3.Session(
        aws_access_key_id='Your_Key_ID',
        aws_secret_access_key='Your_Access_Key',
        region_name='Your_Region'
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket('Your_Bucket_Name')

    for subdir, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=full_path[len(path) + 1:], Body=data)

if __name__ == "__main__":
    upload_files('Your_Filepath')