import pkg_resources

import luigi
import pandas as pd

from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.pipeline.process import BiomDim

LABEL_DICT = {
    'Yes': 1,
    'True': 1,
    'No': 0,
    'False': 0,
    'I do not have this condition': 0,
    'Diagnosed by a medical professional (doctor, physician assistant)': 1
}


def clean_label(label):
    cleaned_label = None

    # map survey results to binary labels
    if label in LABEL_DICT.keys():
        cleaned_label = LABEL_DICT[label]
    return cleaned_label


class Labels(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        paths = [
            'labeled_metadata.csv',
            'label_statistics.csv'
        ]

        outputs = [pkg_resources.resource_filename('american_gut_project.data', p) for p in paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile)

    def run(self):
        metadata = pd.read_csv(self.input().fn)
        ignore_columns = ['index', 'sample_name', 'sample_id']

        label_stats = []
        for label in metadata.columns:

            if label in ignore_columns:
                continue

            num_in_dict = metadata[label].apply(lambda x: x in LABEL_DICT).sum()
            percent_in_label_dict = num_in_dict / len(metadata)
            metadata[label] = metadata[label].apply(clean_label)

            positives = metadata[metadata[label] == 1][label].count()
            negatives = metadata[metadata[label] == 0][label].count()

            label_stats.append({
                'label': label,
                'num_in_dict': num_in_dict,
                'percent_in_label_dict': percent_in_label_dict,
                'positives': positives,
                'negatives': negatives
            })

        label_stats_df = pd.DataFrame(label_stats)
        label_stats_df = label_stats_df.loc[label_stats_df['percent_in_label_dict'].sort_values(ascending=False).index]
        label_stats_df = label_stats_df.reset_index(drop=True)

        metadata_path, stats_path = self.output()[0].fn, self.output()[1].fn
        metadata.to_csv(metadata_path)
        label_stats_df.to_csv(stats_path)


class TrainingData(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    # def output(self):
    #     paths = [
    #         'labeled_metadata.csv',
    #         'label_statistics.csv'
    #     ]
    #
    #     outputs = [pkg_resources.resource_filename('american_gut_project.data', p) for p in paths]
    #     return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [
            FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile),
            BiomDim(aws_profile=self.aws_profile)
        ]

    def run(self):
        metadata, biom = self.input()[0].fn, self.input()[1][0].fn
        biom = pd.read_pickle(biom)
        biom = biom.drop(labels='sample_name', axis=1)
        x = biom.set_index('sample_id')
        x = x.fillna(0)

        print('hi')

if __name__ == '__main__':
    luigi.build([Labels(aws_profile='dse')], local_scheduler=True)
