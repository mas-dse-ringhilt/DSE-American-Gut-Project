import pkg_resources
import pickle

import luigi
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from american_gut_project.pipeline.dataset import BuildTrainingData


class SimpleModel(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    def output(self):
        paths = [
            "{}_simple_model.pkl".format(self.target),
            "{}_simple_model_metrics.txt".format(self.target)
        ]

        outputs = [pkg_resources.resource_filename('american_gut_project.data', p) for p in paths]
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

        metrics_file = self.output()[1].path
        with open(metrics_file, 'w') as f:
            f.write("Label: {}\n".format(self.target))

            predictions = clf.predict(x_train)
            f.write('Training Set: \n')
            f.write(classification_report(y_train, predictions))

            predictions = clf.predict(x_test)
            f.write('Test Set: \n')
            f.write(classification_report(y_test, predictions))


if __name__ == '__main__':
    luigi.build([SimpleModel(aws_profile='dse', target='add_adhd')], local_scheduler=True)
