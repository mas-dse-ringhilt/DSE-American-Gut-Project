import pkg_resources

import boto3
import pandas as pd

BUCKET_NAME = 'dse-cohort4-group1'


def download_file(filename, profile_name='default'):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    s3.Bucket(BUCKET_NAME).download_file(filename, local_file_path)


def upload_file(filename, profile_name='default'):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    s3.meta.client.upload_file(local_file_path, BUCKET_NAME, filename)


def save_dataframe(df, filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)
    df.to_csv(local_file_path)


def load_dataframe(filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)
    pd.read_csv(local_file_path)

if __name__ == '__main__':
    upload_file('test.txt', profile_name='dse')