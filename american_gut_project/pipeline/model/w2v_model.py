import pickle

import luigi
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split

from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.pipeline.embedding.w2v import EmbedBiom
from american_gut_project.paths import paths
from american_gut_project.pipeline.metrics import evaluate


class W2VRandomForest(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    # W2V Parameters
    use_value = luigi.BoolParameter(default=False)
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)

    # Random Forest Parameters
    n_estimators = luigi.IntParameter(default=10)
    max_depth = luigi.IntParameter(default=None)
    min_samples_split = luigi.IntParameter(default=2)
    min_samples_leaf = luigi.IntParameter(default=1)

    def name(self):
        w2v_params = "{}_{}_{}_{}".format(self.use_value, self.min_count, self.size, self.epochs)
        rf_params = "{}_{}_{}_{}".format(self.n_estimators, self.max_depth, self.min_samples_split, self.min_samples_leaf)
        return "{}_w2v_{}_{}".format(self.target, w2v_params, rf_params)

    def param_string(self):
        return "use_value:{} min_count:{} size:{} epochs:{} n_estimators:{} " \
               "max_depth:{} min_samples_split:{} min_samples_leaf:{}".format(self.use_value,
                                                                              self.min_count,
                                                                              self.size,
                                                                              self.epochs,
                                                                              self.n_estimators,
                                                                              self.max_depth,
                                                                              self.min_samples_split,
                                                                              self.min_samples_leaf)

    def output(self):
        output_paths = [
            "{}_rf_model.pkl".format(self.name()),
            "{}_rf_model_metrics.csv".format(self.name())
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

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1
        )
        clf.fit(x_train, y_train)

        model_file = self.output()[0].path
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        name = model_file.split('/')[-1]
        metric_df = evaluate(clf, x_train, x_test, y_train, y_test, name, self.param_string())
        metrics_file = self.output()[1].path

        metric_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    luigi.build([
        W2VRandomForest(aws_profile='dse',
                        target='ibd',
                        use_value=True,
                        min_count=1,
                        size=100,
                        epochs=5,
                        n_estimators=10,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1),

    ], local_scheduler=True, workers=1)
