import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def evaluate(clf, x_train, x_test, y_train, y_test, name, params):
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
        'name': [name],
        'params': [params],
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

    return pd.DataFrame(result_dict)
