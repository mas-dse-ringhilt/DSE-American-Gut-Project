import pickle

import luigi
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.paths import paths
from american_gut_project.pipeline.metrics import evaluate
from american_gut_project.pipeline.embedding.hyperbolic import Hyperbolic
from american_gut_project.pipeline.model.util import balance


class HyperbolicModel(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()
    balance = luigi.BoolParameter(default=False)

    def name(self):
        return "{}_{}_hyperbolic_svc_model.pkl".format(self.target, self.balance)

    def training_data_name(self):
        return "{}_{}_hyperbolic_training_data.pkl".format(self.target, self.balance)

    def output(self):
        output_paths = [
            ("{}_hyperbolic_model.pkl".format(self.target), 'model'),
            ("{}_hyperbolic_model_metrics.csv".format(self.target), 'metrics'),
            (self.training_data_name(), 'training_data'),
        ]

        outputs = [paths.output(p[0], p[1]) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [
            BuildTrainingData(aws_profile=self.aws_profile, target=self.target),
            Hyperbolic(aws_profile=self.aws_profile)
        ]

    def run(self):
        hyperbolic = pd.read_pickle(self.input()[1][1].path)

        training_data = pd.read_pickle(self.input()[0].path)
        training_data = training_data[[self.target]]

        df = hyperbolic.merge(training_data, left_index=True, right_index=True)

        training_data_file = self.output()[2].path
        df.to_pickle(training_data_file)

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        if self.balance:
            sample_df = balance(x_train, y_train, self.target)
            x_train = sample_df.drop(self.target, axis=1)
            y_train = sample_df[self.target]

        clf = SVC()
        clf.fit(x_train, y_train)

        model_file = self.output()[0].path
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        metric_df = evaluate(clf, x_train, x_test, y_train, y_test,
                             self.name(), self.training_data_name(), 'hyperbolic')

        metrics_file = self.output()[1].path
        metric_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    luigi.build([HyperbolicModel(aws_profile='dse', target='feces')], workers=1, local_scheduler=True)
