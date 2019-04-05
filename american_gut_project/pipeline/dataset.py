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


class BuildTrainingData(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        filename = "{}_training_data.pkl".format(self.target)
        local_file_path = pkg_resources.resource_filename('american_gut_project.data', filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return [
            Labels(aws_profile=self.aws_profile),
            BiomDim(aws_profile=self.aws_profile)
        ]

    def run(self):
        labels, biom = self.input()[0][0].fn, self.input()[1][0].fn
        biom = pd.read_pickle(biom)
        labels = pd.read_csv(labels, index_col=0)

        # Duplicate sample ids for some reason
        biom = biom.loc[biom['sample_id'].drop_duplicates().index]
        biom = biom.set_index('sample_id')

        target = labels[['sample_id', self.target]]
        target = target.loc[target[self.target].dropna().index]
        training_data = biom.merge(target, left_on='sample_id', right_on='sample_id')

        training_data.to_pickle(self.output().fn)

if __name__ == '__main__':
    luigi.build([BuildTrainingData(aws_profile='dse', target='add_adhd')], local_scheduler=True)
