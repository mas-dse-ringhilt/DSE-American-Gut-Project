import pkg_resources
import pickle

import luigi
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.paths import paths
from american_gut_project.pipeline.metrics import evaluate


class SimpleModel(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        output_paths = [
            "{}_simple_model.pkl".format(self.target),
            "{}_simple_model_metrics.txt".format(self.target)
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return BuildTrainingData(aws_profile=self.aws_profile, target=self.target)

    def run(self):
        df = pd.read_pickle(self.input().fn)

        df = df.drop(labels='sample_name', axis=1)
        df = df.set_index('sample_id')
        df = df.fillna(0)
        df = df.to_dense()

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        clf = LogisticRegression(penalty='l2', C=1e-3, solver='lbfgs')
        clf.fit(x_train, y_train)

        model_file = self.output()[0].path
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        metric_df = evaluate(clf, x_train, x_test, y_train, y_test, "{}_simple_model.pkl".format(self.target))

        metrics_file = self.output()[1].path
        metric_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    luigi.build([SimpleModel(aws_profile='dse', target='ibd')], local_scheduler=True)
