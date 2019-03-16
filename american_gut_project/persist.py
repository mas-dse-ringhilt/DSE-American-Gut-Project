import pkg_resources
import pickle

import boto3
import pandas as pd
from gensim.models import Word2Vec




def download_file(filename, profile_name):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    s3.Bucket(BUCKET_NAME).download_file(filename, local_file_path)


def upload_file(filename, profile_name):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    session = boto3.session.Session(profile_name=profile_name)
    s3 = session.resource('s3')

    s3.meta.client.upload_file(local_file_path, BUCKET_NAME, filename)


def save_dataframe(df, filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    if filename.endswith('csv'):
        df.to_csv(local_file_path)

    elif filename.endswith('hdf'):
        df.to_hdf(local_file_path, key='df')

    elif filename.endswith('pkl'):
        df.to_pickle(local_file_path)


def save_pickle(obj, filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)
    pickle.dump(obj, open(local_file_path, 'wb'))


def load_pickle(filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)
    return pickle.load(open(local_file_path, 'rb'))


def load_dataframe(filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)

    if filename.endswith('csv'):
        df = pd.read_csv(local_file_path)

    elif filename.endswith('hdf'):
        df = pd.read_hdf(local_file_path, key='df')

    elif filename.endswith('pkl'):
        df = pd.read_pickle(local_file_path)

    return df


def save_w2v_model(model, filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.models', filename)
    model.save(local_file_path)


def load_w2v_model(filename):
    local_file_path = pkg_resources.resource_filename('american_gut_project.models', filename)
    return Word2Vec.load(local_file_path)


if __name__ == '__main__':
    upload_file('agp_only_meta.csv', profile_name='dse')
