import pkg_resources
import pickle

import luigi
import pandas as pd

from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.paths import paths

# Deprecated with new data
# class BiomDim(luigi.Task):
#     aws_profile = luigi.Parameter(default='default')
#
#     def output(self):
#         output_paths = [
#             'biom_dim.pkl',
#             'sequences.pkl'
#         ]
#
#         outputs = [paths.output(p) for p in output_paths]
#         return [luigi.LocalTarget(output) for output in outputs]
#
#     def requires(self):
#         return FetchData(filename='id_clean_biom_v4.pkl', aws_profile=self.aws_profile)
#
#     def run(self):
#         sequence = {}
#         ignore_columns = ['sample_name', 'sample_id']
#         df = pd.read_pickle(self.input().fn)
#
#         new_columns = []
#         for i in range(len(df.columns)):
#
#             column = df.columns[i]
#             if column in ignore_columns:
#                 new_columns.append(column)
#
#             else:
#                 sequence[column] = i
#                 new_columns.append(i)
#
#         df.columns = new_columns
#
#         biom_dim_path, sequences_path = self.output()[0].path, self.output()[1].path
#         df.to_pickle(biom_dim_path)
#         pickle.dump(sequence, open(sequences_path, 'wb'))


class Biom(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        filename = "biom.pkl"
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return FetchData(filename='2.21.rar1000_clean_biom.pkl', aws_profile=self.aws_profile)

    def run(self):
        df = pd.read_pickle(self.input().fn)
        df = df.drop('sample_name', axis=1)
        df = df.loc[df['sample_id'].drop_duplicates().index]
        df = df.set_index('sample_id')

        biom_dim_path = self.output().path
        df.to_pickle(biom_dim_path)


if __name__ == '__main__':
    luigi.build([Biom(aws_profile='dse')], local_scheduler=True)

