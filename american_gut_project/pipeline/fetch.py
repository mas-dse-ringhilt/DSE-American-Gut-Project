import pkg_resources

import boto3
import luigi

from american_gut_project.persist import download_file

BUCKET_NAME = 'dse-cohort4-group1'


class FetchData(luigi.Task):
    filename = luigi.Parameter()
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        local_file_path = pkg_resources.resource_filename('american_gut_project.data', self.filename)
        return luigi.LocalTarget(local_file_path)

    def run(self):
        session = boto3.session.Session(profile_name=self.aws_profile)
        s3 = session.resource('s3')

        s3.Bucket(BUCKET_NAME).download_file(self.filename, self.output().fn)


# def fetch_metadata(profile_name='default'):
#
#
# def fetch_drug_data(profile_name='default'):
#     download_file('drug_data_dense.csv', profile_name)
#
#
# def fetch_biome_data(profile_name='default'):
#     download_file('biom_table.pkl', profile_name)
#
#
# def fetch_all(profile_name='default'):
#     fetch_metadata(profile_name)
#     fetch_drug_data(profile_name)
#     fetch_biome_data(profile_name)
#

if __name__ == '__main__':
    luigi.build([FetchData(filename='agp_only_meta.csv', aws_profile='dse')], local_scheduler=True)

