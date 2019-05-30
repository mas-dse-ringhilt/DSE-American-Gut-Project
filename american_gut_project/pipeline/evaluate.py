import os

import luigi
import pandas as pd
from sklearn.decomposition import PCA

from american_gut_project.pipeline.model.w2v_model import W2VRandomForest, W2VLogisticRegression, W2VXGBoost
from american_gut_project.pipeline.model.simple_model import SimpleModel
from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.paths import paths


class Metrics(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def requires(self):
        task_list = []

        # simple model
        for balance in [True, False]:
            simple_model = SimpleModel(aws_profile=self.aws_profile, target=self.target, balance=balance)
            task_list.append(simple_model)
        #
        # for balance in [True, False]:
        #     for size in [80, 100, 120]:
        #         for epochs in [5]:
        #             for n_estimators in [10, 12]:
        #                 for min_samples_split in [2]:
        #                     for min_samples_leaf in [1]:
        #                         for max_depth in [None]:
        #                             for min_count in [1]:
        #                                 task = W2VRandomForest(aws_profile=self.aws_profile,
        #                                                        target=self.target,
        #                                                        use_value=True,
        #                                                        balance=balance,
        #                                                        min_count=min_count,
        #                                                        size=size,
        #                                                        epochs=epochs,
        #                                                        n_estimators=n_estimators,
        #                                                        max_depth=max_depth,
        #                                                        min_samples_split=min_samples_split,
        #                                                        min_samples_leaf=min_samples_leaf)
        #                                 task_list.append(task)

        for balance in [True]:
            for size in [100, 120]:
                for epochs in [5]:
                    task = W2VLogisticRegression(aws_profile=self.aws_profile,
                                                 target=self.target,
                                                 use_value=True,
                                                 balance=balance,
                                                 min_count=1,
                                                 size=size,
                                                 epochs=epochs)
                    task_list.append(task)

        for balance in [True]:
            for size in [60, 80, 100, 120]:
                for epochs in [10, 15]:
                    for n_estimators in [130, 150, 200]:
                        for max_depth in [2, 3, 4]:
                            for min_count in [1]:
                                for scale_pos_weight in [True, False]:
                                    for alpha_diversity in [True, False]:
                                        task = W2VXGBoost(aws_profile=self.aws_profile,
                                                          target=self.target,
                                                          use_value=True,
                                                          balance=balance,
                                                          alpha_diversity=alpha_diversity,
                                                          min_count=min_count,
                                                          size=size,
                                                          epochs=epochs,
                                                          n_estimators=n_estimators,
                                                          max_depth=max_depth,
                                                          scale_pos_weight=scale_pos_weight)
                                        task_list.append(task)
        return task_list

    def run(self):
        df_list = []
        for metric_file in self.input():
            metric_file = metric_file[1]
            df = pd.read_csv(metric_file.fn)
            df_list.append(df)


class Analysis(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    models = ['w2v', 'simple']

    def output(self):
        output_paths = []
        for model in self.models:
            output_paths.append("{}_pca.csv".format(model))

        outputs = [paths.output(p, 'pca') for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile)]


    def run(self):
        metadata = pd.read_csv(self.input()[0].fn, index_col=0)
        metadata['sample_id'] = metadata['sample_id'].astype(str)
        metadata = metadata.set_index('sample_id')
        metadata = metadata[['env_material']]

        file_path = paths.output('', 'metrics')

        file_list = os.listdir(file_path)

        df_list = []
        for f in file_list:
            if f.startswith(self.target):
                df = pd.read_csv(os.path.join(file_path, f))
                df_list.append(df)

        metrics = pd.concat(df_list, ignore_index=True)

        for i, model in enumerate(self.models):
            df = metrics[metrics['embedding'] == model]
            best = df.loc[df['test_f1_score'].idxmax()]
            best_training_data = best['training_data_name']
            training_data_path = paths.output(best_training_data, 'training_data')

            training_data = pd.read_pickle(training_data_path)

            print('Training data shape', training_data.shape)

            idx = training_data.index
            X = training_data.drop(target, axis=1)

            pca = PCA(n_components=3)
            transformed = pd.DataFrame(pca.fit_transform(X), index=idx)
            transformed = transformed.merge(metadata, left_index=True, right_index=True)
            print("{} Explained Variance".format(model), pca.explained_variance_ratio_)

            output_path = self.output()[i].path
            transformed.to_csv(output_path)



if __name__ == '__main__':
    target = 'feces'
    # output_file = "{}_metrics.csv".format(target)
    # file_path = paths.output(output_file)
    #
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    #
    # luigi.build([
    #     Metrics(aws_profile='dse', target=target),
    # ], workers=4, local_scheduler=True)

    luigi.build([
        Analysis(aws_profile='dse', target=target),
    ], workers=4, local_scheduler=True)
