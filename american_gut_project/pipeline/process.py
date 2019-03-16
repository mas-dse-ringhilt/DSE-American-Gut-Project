import pkg_resources
import pickle

import luigi
import pandas as pd

from american_gut_project.pipeline.fetch import FetchData


class BiomDim(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        paths = [
            'biom_dim.pkl',
            'sequences.pkl'
        ]

        outputs = [pkg_resources.resource_filename('american_gut_project.data', p) for p in paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return FetchData(filename='id_clean_biom_v4.pkl', aws_profile=self.aws_profile)

    def run(self):
        sequence = {}
        # ignore_columns = ['sample_name', 'sample_id']
        df = pd.read_pickle(self.input().fn)

        for i in range(len(df.columns)):
            sequence[df.columns[i]] = i

        new_columns = []
        for col in df.columns:
            new_columns.append(sequence[col])

        df.columns = new_columns

        biom_dim_path, sequences_path = self.output()[0].fn, self.output()[1].fn
        df.to_pickle(biom_dim_path)
        pickle.dump(sequence, open(sequences_path, 'wb'))


if __name__ == '__main__':
    luigi.build([BiomDim(aws_profile='dse')], local_scheduler=True)

