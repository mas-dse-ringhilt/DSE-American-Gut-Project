import os

import luigi
import pandas as pd
from sklearn.decomposition import PCA

from american_gut_project.pipeline.model.w2v_model import W2VLogisticRegression
from american_gut_project.pipeline.model.simple_model import SimpleModel
from american_gut_project.pipeline.model.hyperbolic import HyperbolicModel
from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.paths import paths


class Metrics(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        output_paths = [
            "{}_combined_metrics.csv".format(self.target)
        ]
        outputs = [paths.output(p, 'metrics') for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        task_list = []

        # simple model
        simple_model = SimpleModel(aws_profile=self.aws_profile, target=self.target)
        task_list.append(simple_model)

        # hyperbolic
        hyperbolic_model = HyperbolicModel(aws_profile=self.aws_profile, target=self.target)
        task_list.append(hyperbolic_model)

        # w2v
        w2v_model = W2VLogisticRegression(aws_profile=self.aws_profile,
                                          target=self.target,
                                          use_value=True,
                                          min_count=1,
                                          size=80,
                                          epochs=10)
        task_list.append(w2v_model)

        return task_list

    def run(self):
        df_list = []
        for metric_file in self.input():
            metric_file = metric_file[1]
            df = pd.read_csv(metric_file.fn)
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        output_path = self.output()[0].path
        df.to_csv(output_path, index=None)


class Analysis(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    models = ['w2v', 'simple', 'hyperbolic']

    def output(self):
        output_paths = []
        for model in self.models:
            output_paths.append("{}_pca.csv".format(model))

        outputs = [paths.output(p, 'pca') for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile),
                Metrics(aws_profile=self.aws_profile, target=self.target)]

    def run(self):
        metadata = pd.read_csv(self.input()[0].fn, index_col=0)
        metadata['sample_id'] = metadata['sample_id'].astype(str)
        metadata = metadata.set_index('sample_id')
        metadata = metadata[['env_material']]

        metrics = pd.read_csv(self.input()[1][0].fn)

        for i, model in enumerate(self.models):
            df = metrics[metrics['embedding'] == model]
            best = df.loc[df['test_f1_score'].idxmax()]
            best_training_data = best['training_data_name']
            training_data_path = paths.output(best_training_data, 'training_data')

            training_data = pd.read_pickle(training_data_path)

            print('Training data shape', training_data.shape)

            idx = training_data.index
            X = training_data.drop(self.target, axis=1)

            pca = PCA(n_components=3)
            transformed = pd.DataFrame(pca.fit_transform(X), index=idx)
            transformed = transformed.merge(metadata, left_index=True, right_index=True)
            print("{} Explained Variance".format(model), pca.explained_variance_ratio_)

            output_path = self.output()[i].path
            transformed.to_csv(output_path)

if __name__ == '__main__':
    target = 'body_site_target'
    luigi.build([
        Analysis(aws_profile='dse', target=target),
    ], workers=4, local_scheduler=True)
