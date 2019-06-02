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


def compute_alpha_diversity(row):
    return row.dropna().count()


def split_sample_id(sample_name):
    sample_id = sample_name.split('.')[1]
    try:
        return str(int(sample_id))
    except ValueError:
        return None


class AlphaDiversity(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        filename = "alpha_diversity.pkl"
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return Biom(aws_profile=self.aws_profile)

    def run(self):
        df = pd.read_pickle(self.input().fn)

        alpha_diversity = pd.DataFrame(df.apply(compute_alpha_diversity, axis=1), columns=['alpha_diversity'], dtype=int)
        alpha_diversity_path = self.output().path
        alpha_diversity.to_pickle(alpha_diversity_path)


class Biom(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        filename = "biom.pkl"
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return [
            FetchData(filename='4.10.rar1000.biom_data.pkl', aws_profile=self.aws_profile),
            FetchData(filename='sample_ids_with_host.csv', aws_profile=self.aws_profile)
        ]

    def run(self):
        biom = pd.read_pickle(self.input()[0].fn)

        if 'sample_id' not in biom.columns:
            sample_ids = []
            for i in range(len(biom)):
                sample_name = biom['sample_name'].astype(str).iloc[i]

                sample_id = split_sample_id(sample_name)
                sample_ids.append(sample_id)

            biom['sample_id'] = sample_ids
            biom = biom.loc[biom['sample_id'].dropna().index]

        biom = biom.drop('sample_name', axis=1)
        biom = biom.loc[biom['sample_id'].drop_duplicates().index]
        biom = biom.set_index('sample_id')

        # samples = pd.read_csv(self.input()[1].fn)
        # samples = samples.loc[samples['host_subject_id'].dropna().drop_duplicates().index]
        # sample_ids = samples['sample_id']
        #
        # biom = biom.loc[sample_ids]
        biom_dim_path = self.output().path
        biom.to_pickle(biom_dim_path)


if __name__ == '__main__':
    luigi.build([Biom(aws_profile='dse')], local_scheduler=True)

