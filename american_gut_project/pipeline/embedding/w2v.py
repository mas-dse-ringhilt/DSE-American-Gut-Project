import pkg_resources
import pickle
from random import shuffle

import gensim
import luigi
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from american_gut_project.pipeline.dataset import BuildTrainingData
from american_gut_project.pipeline.process import BiomDim

# def build_microbiome_embeddings():
#     # pull table from local file
#     samples = load_dataframe('biom_table.pkl')
#
#     # replace nans with zeros
#     samples = samples.fillna(0)
#
#     sentences = []
#     for sample in range(len(samples) - 1):
#         try:
#             sentence = list(samples[sample][samples[sample] > 0].index)
#             sentences.append([str(x) for x in sentence])
#         except KeyError:
#             pass
#
#     # shuffle each sample around and append it to the training data
#     augmentation_constant = 2
#     generated_sentences = []
#     for sentence in sentences:
#         for augmentation in range(augmentation_constant):
#             shuffle(sentence)
#             generated_sentences.append(sentence)
#
#     model = gensim.models.Word2Vec(generated_sentences, size=100, min_count=2, window=100, workers=4, sg=0)
#     model.train(generated_sentences, total_examples=len(sentences), epochs=5)
#     save_w2v_model(model, 'microbiome_w2v.model')
#


def build_sentences(df):
    sentences = []
    for i in range(len(df)):
        sentence = []
        samples_with_values = df.iloc[i][df.iloc[i] > 0]

        for j in range(len(samples_with_values)):
            sample = [str(samples_with_values.index[j])]
            value = samples_with_values.iloc[j]

            sentence += sample * int(value)

        sentences.append(sentence)

    return sentences


def apply_model(sentences, model):

    sample_vectors = []
    for sentence in sentences:

        sentence_vector = []
        for word in sentence:
            word_vector = model.wv[word]
            sentence_vector.append(word_vector)

        mean_vector = np.mean(sentence_vector, axis=0)
        sample_vectors.append(mean_vector)

    return sample_vectors

# TODO
# class Sentences(luigi.Task):
#     aws_profile = luigi.Parameter(default='default')
#
#     def output(self):
#
#         sentences_path = pkg_resources.resource_filename('american_gut_project.data', 'sentences.pkl')
#
#         outputs = [w2v_model_path, w2v_biom_path]
#         return [luigi.LocalTarget(output) for output in outputs]
#
#     def requires(self):
#         return BiomDim(aws_profile=self.aws_profile)
#
#     def run(self):
#         df = pd.read_pickle(self.input()[0].fn)
#         df = df.drop('sample_name', axis=1)
#         df = df.loc[df['sample_id'].drop_duplicates().index]
#         df = df.set_index('sample_id')
#
#         # replace nans with zeros
#         df = df.fillna(0)
#
#         print('building sentences')
#         sentences = build_sentences(df)
#
#         print('training')
#         model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=100, workers=4, sg=0)
#         model.train(sentences, total_examples=len(sentences), epochs=5)
#         model.save(self.output()[0].path)
#
#         sample_vectors = apply_model(sentences, model)
#         sample_df = pd.DataFrame(data=sample_vectors, index=df.index)
#         sample_df.to_pickle(self.output()[1].path)

class W2V(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        paths = [
            'w2v.model',
            'w2v_biom.pkl'
        ]

        w2v_model_path = pkg_resources.resource_filename('american_gut_project.model', paths[0])
        w2v_biom_path = pkg_resources.resource_filename('american_gut_project.data', paths[1])

        outputs = [w2v_model_path, w2v_biom_path]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return BiomDim(aws_profile=self.aws_profile)

    def run(self):
        df = pd.read_pickle(self.input()[0].fn)
        df = df.drop('sample_name', axis=1)
        df = df.loc[df['sample_id'].drop_duplicates().index]
        df = df.set_index('sample_id')

        # replace nans with zeros
        df = df.fillna(0)

        print('building sentences')
        sentences = build_sentences(df)

        print('training')
        model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=100, workers=4, sg=0)
        model.train(sentences, total_examples=len(sentences), epochs=5)
        model.save(self.output()[0].path)

        sample_vectors = apply_model(sentences, model)
        sample_df = pd.DataFrame(data=sample_vectors, index=df.index)
        sample_df.to_pickle(self.output()[1].path)


if __name__ == '__main__':
    luigi.build([W2V(aws_profile='dse')], local_scheduler=True)
