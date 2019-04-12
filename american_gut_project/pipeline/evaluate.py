import pkg_resources
import pickle

import luigi
import pandas as pd


from american_gut_project.pipeline.model.w2v_model import W2VRandomForest
from american_gut_project.pipeline.model.simple_model import SimpleModel

from american_gut_project.paths import paths


class Metrics(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        output_paths = [
            "{}_metrics.csv".format(self.target),
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        task_list = []

        # simple model
        simple_model = SimpleModel(aws_profile=self.aws_profile, target=self.target)
        task_list.append(simple_model)

        # w2v_random_forest
        # for min_count in range(3):
        #     for size in range(25, 300, 25):
        #         for epochs in range(5, 25, 5):
        #             for n_estimators in [10, 20, 30]:
        #                 for max_depth in [None, 10, 20, 30]:
        #                     for min_samples_split in [2, 4, 6, 8]:
        #                         for min_samples_leaf in [1, 2, 5, 10]:
        #                             task = W2VRandomForest(aws_profile=self.aws_profile,
        #                                                    target=self.target,
        #                                                    use_value=True,
        #                                                    min_count=min_count,
        #                                                    size=size,
        #                                                    epochs=epochs,
        #                                                    n_estimators=n_estimators,
        #                                                    max_depth=max_depth,
        #                                                    min_samples_split=min_samples_split,
        #                                                    min_samples_leaf=min_samples_leaf)
        #                             task_list.append(task)

        for size in [50, 75, 100, 125]:
            for epochs in [5, 10, 15]:
                for n_estimators in [10, 12, 20]:
                    for min_samples_split in [2, 3, 4]:
                        for min_samples_leaf in [1, 2, 3]:
                            task = W2VRandomForest(aws_profile=self.aws_profile,
                                                   target=self.target,
                                                   use_value=True,
                                                   min_count=1,
                                                   size=size,
                                                   epochs=epochs,
                                                   n_estimators=n_estimators,
                                                   max_depth=None,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf)
                            task_list.append(task)
        return task_list

    def run(self):

        df_list = []
        for metric_file in self.input():
            metric_file = metric_file[1]
            df = pd.read_csv(metric_file.fn)
            df_list.append(df)

        metrics = pd.concat(df_list, ignore_index=True)
        output_path = self.output()[0].path
        metrics.to_csv(output_path, index=False)


if __name__ == '__main__':
    luigi.build([
        Metrics(aws_profile='dse', target='ibd'),
    ], local_scheduler=True, workers=5)
