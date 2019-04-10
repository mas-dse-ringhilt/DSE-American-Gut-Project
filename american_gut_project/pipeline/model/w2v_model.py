import pkg_resources
import pickle

import luigi
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.pipeline.embedding.w2v import EmbedBiom
from american_gut_project.paths import paths


class W2VModel(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()
    use_value = luigi.BoolParameter(default=False)
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)


    def output(self):
        output_paths = [
            "{}_w2v_{}_{}_{}_{}_model.pkl".format(self.target, self.use_value, self.min_count, self.size, self.epochs),
            "{}_w2v_{}_{}_{}_{}_model_metrics.csv".format(self.target, self.use_value, self.min_count, self.size, self.epochs)
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [
            BuildTrainingData(aws_profile=self.aws_profile, target=self.target),
            EmbedBiom(aws_profile=self.aws_profile, use_value=self.use_value, min_count=self.min_count, size=self.size, epochs=self.epochs)
        ]

    def run(self):
        biom = pd.read_pickle(self.input()[0].fn)

        biom = biom.drop(labels='sample_name', axis=1)
        biom = biom.set_index('sample_id')

        w2v = pd.read_pickle(self.input()[1].fn)

        target = biom[[self.target]]
        df = w2v.merge(target, left_index=True, right_index=True, how='inner')

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)

        model_file = self.output()[0].path
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        predictions = clf.predict(x_train)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, predictions).ravel()
        train_accuracy = accuracy_score(y_train, predictions)
        train_precision = precision_score(y_train, predictions)
        train_recall = recall_score(y_train, predictions)

        predictions = clf.predict(x_test)
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, predictions).ravel()
        test_accuracy = accuracy_score(y_test, predictions)
        test_precision = precision_score(y_test, predictions)
        test_recall = recall_score(y_test, predictions)

        result_dict = {
            'name': ["{}_w2v_{}_{}_{}_{}_model.pkl".format(self.target, self.use_value, self.min_count, self.size, self.epochs)],
            'train_true_negative': [train_tn],
            'train_false_positive': [train_fp],
            'train_false_negative': [train_fn],
            'train_true_positive': [train_tp],
            'train_accuracy': [train_accuracy],
            'train_precision': [train_precision],
            'train_recall': [train_recall],

            'test_true_negative': [test_tn],
            'test_false_positive': [test_fp],
            'test_false_negative': [test_fn],
            'test_true_positive': [test_tp],
            'test_accuracy': [test_accuracy],
            'test_precision': [test_precision],
            'test_recall': [test_recall],

        }

        metrics_file = self.output()[1].path
        metric_df = pd.DataFrame(result_dict)
        metric_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    luigi.build([
        W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=50, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=200, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=300, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=10),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=15),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=20),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=1, size=100, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=50, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=200, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=300, epochs=5),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=100, epochs=10),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=100, epochs=15),
        # W2VModel(aws_profile='dse', target='ibd', use_value=True, min_count=2, size=100, epochs=20)
    ], local_scheduler=True, workers=3)
