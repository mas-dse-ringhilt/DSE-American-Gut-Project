import pickle

import luigi
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

from american_gut_project.paths import paths
from american_gut_project.pipeline.process import AlphaDiversity
from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.pipeline.embedding.w2v import EmbedBiom
from american_gut_project.pipeline.metrics import evaluate
from american_gut_project.pipeline.model.util import balance


class W2VRandomForest(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    # Data Parameters
    balance = luigi.BoolParameter(default=False)

    # Feature Parameters
    alpha_diversity = luigi.BoolParameter(default=False)

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
        data_params = "{}".format(self.balance)
        feature_params = "{}".format(self.alpha_diversity)
        w2v_params = "{}_{}_{}_{}".format(self.use_value, self.min_count, self.size, self.epochs)
        rf_params = "{}_{}_{}_{}".format(self.n_estimators, self.max_depth, self.min_samples_split, self.min_samples_leaf)
        return "{}_w2v_{}_{}_{}_{}".format(self.target, data_params, feature_params, w2v_params, rf_params)

    def param_string(self):
        return "alpha_diversity:{} balance: {} use_value:{} min_count:{} size:{} epochs:{} n_estimators:{} " \
               "max_depth:{} min_samples_split:{} min_samples_leaf:{}".format(self.alpha_diversity,
                                                                              self.balance,
                                                                              self.use_value,
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
        task_list = [
            BuildTrainingData(aws_profile=self.aws_profile, target=self.target),
            EmbedBiom(aws_profile=self.aws_profile, use_value=self.use_value, min_count=self.min_count,
                      size=self.size, epochs=self.epochs)
        ]

        if self.alpha_diversity:
            task_list.append(AlphaDiversity(aws_profile=self.aws_profile))

        return task_list

    def run(self):
        biom = pd.read_pickle(self.input()[0].fn)
        w2v = pd.read_pickle(self.input()[1].fn)

        target = biom[[self.target]]
        df = w2v.merge(target, left_index=True, right_index=True, how='inner')

        if self.alpha_diversity:
            alpha = pd.read_pickle(self.input()[2].fn)
            df = df.merge(alpha, left_index=True, right_index=True, how='inner')

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        if self.balance:
            sample_df = balance(x_train, y_train, self.target)
            x_train = sample_df.drop(self.target, axis=1)
            y_train = sample_df[self.target]

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


class W2VLogisticRegression(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    # Data Parameters
    balance = luigi.BoolParameter(default=False)

    # W2V Parameters
    use_value = luigi.BoolParameter(default=False)
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)

    def name(self):
        data_params = "{}".format(self.balance)
        w2v_params = "{}_{}_{}_{}".format(self.use_value, self.min_count, self.size, self.epochs)
        lr_params = ""
        return "{}_w2v_{}_{}_{}".format(self.target, data_params, w2v_params, lr_params)

    def param_string(self):
        return "use_value:{} balance:{} min_count:{} size:{} epochs:{}".format(self.use_value,
                                                                               self.balance,
                                                                               self.min_count,
                                                                               self.size,
                                                                               self.epochs)

    def output(self):
        output_paths = [
            "{}_lr_model.pkl".format(self.name()),
            "{}_lr_model_metrics.csv".format(self.name())
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
        w2v = pd.read_pickle(self.input()[1].fn)

        target = biom[[self.target]]
        df = w2v.merge(target, left_index=True, right_index=True, how='inner')

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        if self.balance:
            sample_df = balance(x_train, y_train, self.target)
            x_train = sample_df.drop(self.target, axis=1)
            y_train = sample_df[self.target]

        clf = LogisticRegression(
            n_jobs=-1,
            penalty='l2',
            C=1e-3,
            solver='lbfgs'
        )
        clf.fit(x_train, y_train)

        model_file = self.output()[0].path
        with open(model_file, 'wb') as f:
            pickle.dump(clf, f)

        name = model_file.split('/')[-1]
        metric_df = evaluate(clf, x_train, x_test, y_train, y_test, name, self.param_string())
        metrics_file = self.output()[1].path

        metric_df.to_csv(metrics_file, index=False)


class W2VXGBoost(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    target = luigi.Parameter()

    # Data Parameters
    balance = luigi.BoolParameter(default=False)

    # Feature Parameters
    alpha_diversity = luigi.BoolParameter(default=False)

    # W2V Parameters
    use_value = luigi.BoolParameter(default=False)
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)

    # XGBoost Parameters
    n_estimators = luigi.IntParameter(default=100)
    max_depth = luigi.IntParameter(default=3)
    scale_pos_weight = luigi.BoolParameter(default=False)

    def name(self):
        data_params = "{}".format(self.balance)
        feature_params = "{}".format(self.alpha_diversity)
        w2v_params = "{}_{}_{}_{}".format(self.use_value, self.min_count, self.size, self.epochs)
        xgb_params = "{}_{}_{}".format(self.n_estimators, self.max_depth, self.scale_pos_weight)
        return "{}_w2v_{}_{}_{}_{}".format(self.target, data_params, feature_params, w2v_params, xgb_params)

    def param_string(self):
        return "alpha_diversity:{} balance:{}, use_value:{} min_count:{} size:{} epochs:{} " \
               "n_estimators:{} max_depth:{} scale_pos_weight:{}".format(self.alpha_diversity,
                                                                         self.balance,
                                                                         self.use_value,
                                                                         self.min_count,
                                                                         self.size,
                                                                         self.epochs,
                                                                         self.n_estimators,
                                                                         self.max_depth,
                                                                         self.scale_pos_weight)

    def output(self):
        output_paths = [
            "{}_xgb_model.pkl".format(self.name()),
            "{}_xgb_model_metrics.csv".format(self.name())
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        task_list = [
            BuildTrainingData(aws_profile=self.aws_profile, target=self.target),
            EmbedBiom(aws_profile=self.aws_profile, use_value=self.use_value, min_count=self.min_count,
                      size=self.size, epochs=self.epochs)
        ]

        if self.alpha_diversity:
            task_list.append(AlphaDiversity(aws_profile=self.aws_profile))

        return task_list

    def run(self):
        biom = pd.read_pickle(self.input()[0].fn)
        w2v = pd.read_pickle(self.input()[1].fn)

        target = biom[[self.target]]
        df = w2v.merge(target, left_index=True, right_index=True, how='inner')

        if self.alpha_diversity:
            alpha = pd.read_pickle(self.input()[2].fn)
            df = df.merge(alpha, left_index=True, right_index=True, how='inner')

        X = df.drop(self.target, axis=1)

        df_dict = {}
        for column in X.columns:
            values = []
            for value in X[column].values:
                values.append(value)

            df_dict[column] = values

        new_X = pd.DataFrame(data=df_dict, index=X.index)
        y = df[self.target]

        x_train, x_test, y_train, y_test = train_test_split(new_X, y, test_size=0.33, random_state=1)

        if self.balance:
            sample_df = balance(x_train, y_train, self.target)
            x_train = sample_df.drop(self.target, axis=1)
            y_train = sample_df[self.target]

        if self.scale_pos_weight:
            ratio = float(np.sum(df[self.target] == 0)) / np.sum(df[self.target] == 1)
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                scale_pos_weight=ratio
            )

        else:
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth
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
        W2VXGBoost(aws_profile='dse',
                   target='ibd',
                   alpha_diversity=True,
                   balance=True,
                   use_value=True,
                   min_count=1,
                   size=100,
                   epochs=5),

    ], local_scheduler=True, workers=1)
