from american_gut_project.persist import load_dataframe, download_file, save_dataframe
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def download_files(files, profile_name):
    # download files required for building model
    for file in files:
        download_file(file, profile_name)


def clean_labels(label):
    # map survey results to binary labels
    label_dict = {'Yes': 1,
                  'True': 1,
                  'No': 0,
                  'False': 0,
                  'I do not have this condition': 0,
                  'Diagnosed by a medical professional (doctor, physician assistant)': 1}
    if label in label_dict.keys():
        return label_dict[label]
    else:
        return None


def build_label(file, label, sample_type):
    """
    Read AGP metadata file and collect the label for the classifier. Sample
    type can be specified for sample collection site.
    """
    # read data
    metadata = load_dataframe(file)
    # select sample type and label
    metadata = metadata[metadata['sample_type'] == sample_type]
    y = metadata[['sample_id', label]].set_index('sample_id')
    # map to binary labels and drop nans
    y = y[label].apply(clean_labels)
    y = y.dropna(how='any', axis=0)
    return pd.DataFrame(y), label


def build_features(file):
    # takes in a biom table and returns df with sample_id as index
    biom = load_dataframe(file)
    biom = biom.drop(labels='sample_name', axis=1)
    x = biom.set_index('sample_id')
    x = x.fillna(0)
    return x


def join_and_split(x, y, label):
    # join features and label
    data = x.merge(y, how='inner', left_index=True, right_index=True)
    data = pd.DataFrame(data)
    y_joined = pd.DataFrame(data[label])
    x_joined = data.drop(labels=label, axis=1)

    save_dataframe(data, "{}_training_df.csv".format(label))

    # create train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_joined, y_joined, test_size=0.33, random_state=1)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    # train logistic regression model
    clf = LogisticRegression(penalty='l2', C=1e-3, solver='lbfgs')
    clf.fit(x_train, y_train.values.ravel())
    y_train_pred = clf.predict(x_train)
    print('Training Set: \n')
    print(classification_report(y_train, y_train_pred))
    return clf


def test_model(clf, x_test, y_test):
    y_test_pred = clf.predict(x_test)
    print('Test Set: \n')
    print(classification_report(y_test, y_test_pred))


def save_model():
    # save model to S3 model bucket
    return


def main():
    files = ['id_clean_biom_v4.pkl', 'agp_only_meta.csv']
    try:
        y, label = build_label('agp_only_meta.csv', 'ibd', 'Stool')
        x = build_features('id_clean_biom_v4.pkl')
    except FileNotFoundError:
        print('Local files not found. Downloading from S3')
        download_files(files, profile_name='dse')
        y, label = build_label('agp_only_meta.csv', 'ibd', 'Stool')
        x = build_features('id_clean_biom_v4.pkl')
    # join data + test train split
    # x_train, x_test, y_train, y_test = join_and_split(x, y, label)
    #
    # # train model and report results
    # clf = train_model(x_train, y_train)
    # test_model(clf, x_test, y_test)


if __name__ == '__main__':
    main()
