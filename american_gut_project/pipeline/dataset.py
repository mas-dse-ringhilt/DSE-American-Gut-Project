import luigi
import pandas as pd

from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.pipeline.process import Biom
from american_gut_project.paths import paths

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


class BodySite(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        output_paths = [
            'body_site.csv'
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return FetchData(filename='agp_only_meta.csv',
                         aws_profile=self.aws_profile)

    def run(self):
        metadata = pd.read_csv(self.input().fn, index_col=0)
        metadata = metadata.drop('sample_name', axis=1)
        metadata = metadata.set_index('sample_id')

        possible_sites = ['feces', 'saliva', 'sebum']

        site_dict = {}
        for site in possible_sites:
            site_dict[site] = metadata['env_material'].apply(lambda x: 1 if x == site else 0)

        site_dict['body_site_target'] = metadata['env_material'].apply(lambda x: x if x in possible_sites else None)

        site_df = pd.DataFrame(site_dict)
        site_path = self.output()[0].path
        site_df.to_csv(site_path)


class Labels(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        output_paths = [
            'labeled_metadata.csv',
            'label_statistics.csv'
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile),
                BodySite(aws_profile=self.aws_profile)]

    def run(self):
        metadata = pd.read_csv(self.input()[0].fn, index_col=0)
        metadata = metadata.drop('sample_name', axis=1)
        metadata = metadata.set_index('sample_id')
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

        body_site_df = pd.read_csv(self.input()[1][0].fn, index_col=0)
        metadata = metadata.merge(body_site_df, left_index=True, right_index=True)

        metadata_path, stats_path = self.output()[0].path, self.output()[1].path
        metadata.to_csv(metadata_path)
        label_stats_df.to_csv(stats_path)


class BuildTrainingData(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        filename = "{}_training_data.pkl".format(self.target)
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return [
            Labels(aws_profile=self.aws_profile),
            Biom(aws_profile=self.aws_profile)
        ]

    def run(self):
        labels, biom = self.input()[0][0].fn, self.input()[1].fn
        biom = pd.read_pickle(biom)
        labels = pd.read_csv(labels)
        labels['sample_id'] = labels['sample_id'].astype(str)
        labels = labels.set_index('sample_id')

        target = labels[[self.target]]
        target = target.loc[target[self.target].dropna().index]
        training_data = biom.merge(target, left_index=True, right_index=True)

        training_data.to_pickle(self.output().path)

if __name__ == '__main__':
    # luigi.build([Labels(aws_profile='dse')], local_scheduler=True)

    luigi.build([BuildTrainingData(aws_profile='dse', target='body_site_target')], local_scheduler=True)
