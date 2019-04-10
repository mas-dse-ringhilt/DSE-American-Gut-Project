import pkg_resources
import datetime

import gensim
import luigi
import pandas as pd
import numpy as np

from american_gut_project.pipeline.process import BiomDim
from american_gut_project.paths import paths

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


class SubSentence(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    min_value = luigi.IntParameter()
    max_value = luigi.IntParameter()
    use_value = luigi.BoolParameter(default='False')

    def output(self):
        filename = "sentences_{}_{}_{}.cor".format(self.min_value, self.max_value, self.use_value)
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return BiomDim(aws_profile=self.aws_profile)

    def run(self):
        df = pd.read_pickle(self.input()[0].fn)
        df = df.drop('sample_name', axis=1)
        df = df.loc[df['sample_id'].drop_duplicates().index]
        df = df.set_index('sample_id')

        df = df[self.min_value:self.max_value]

        print('building sentences')
        with open(self.output().path, 'w') as f:
            build_sentences(df, f, use_value=self.use_value)


def build_sentences(df, file_handler, use_value=False):
    start_time = datetime.datetime.now()
    if use_value:
        print('using value')

    for i in range(len(df)):
        if i % 1000 == 0:
            marker = datetime.datetime.now()
            print(marker - start_time)
            print(i)

        if use_value:
            print(i)
            samples_with_values = df.iloc[i].dropna()

            samples = []
            for j in range(len(samples_with_values)):

                sample = [str(samples_with_values.index[j])]
                value = samples_with_values.iloc[j]

                sample = sample * int(value)
                samples += sample

            sentence = ' '.join(samples)
            file_handler.write(sentence + '\n')

        else:
            print(i)
            sentence = ' '.join(df.iloc[i].dropna().index.astype(str))
            file_handler.write(sentence + '\n')


class Sentences(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    use_value = luigi.BoolParameter(default='False')

    def output(self):
        filename = "sentences_{}.cor".format(self.use_value)
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        task_list = []
        for i in range(0, 15000, 1000):
            task = SubSentence(aws_profile=self.aws_profile,
                               use_value=self.use_value,
                               min_value=i,
                               max_value=i+1000)

            task_list.append(task)
        return task_list

    def run(self):
        with open(self.output().path, 'w') as outfile:
            for input_file in self.input():
                with open(input_file.fn) as f:
                    sentences = f.read()
                    outfile.write(sentences)


class TrainW2V(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    use_value = luigi.BoolParameter(default='False')
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)

    def output(self):
        filename = "w2v_{}_{}_{}_{}.model".format(self.use_value, self.min_count, self.size, self.epochs)
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return Sentences(aws_profile=self.aws_profile, use_value=self.use_value)

    def run(self):
        sentences = self.input().path
        model = gensim.models.Word2Vec(min_count=self.min_count, size=self.size, workers=4)
        model.build_vocab(corpus_file=sentences)
        model.train(corpus_file=sentences, total_examples=model.corpus_count, total_words=len(model.wv.index2entity), epochs=self.epochs)
        model.save(self.output().path)


class EmbedBiom(luigi.Task):
    aws_profile = luigi.Parameter(default='default')
    use_value = luigi.BoolParameter(default='False')
    min_count = luigi.IntParameter(default=1)
    size = luigi.IntParameter(default=100)
    epochs = luigi.IntParameter(default=5)

    def output(self):
        filename = "biom_dim_w2v_{}_{}_{}_{}.pkl".format(self.use_value, self.min_count, self.size, self.epochs)
        local_file_path = paths.output(filename)
        return luigi.LocalTarget(local_file_path)

    def requires(self):
        return [
            BiomDim(aws_profile=self.aws_profile),
            TrainW2V(aws_profile=self.aws_profile, use_value=self.use_value, min_count=self.min_count, size=self.size, epochs=self.epochs)
        ]

    def run(self):
        model = gensim.models.Word2Vec.load(self.input()[1].fn)

        def embed(row):
            sentence = row.dropna().index.astype(str)

            word_vectors = []
            for word in sentence:
                if word in model.wv:
                    word_vector = model.wv[word]
                    word_vectors.append(word_vector)

            return pd.Series(np.mean(word_vectors, axis=0))

        df = pd.read_pickle(self.input()[0][0].fn)
        df = df.drop('sample_name', axis=1)
        df = df.loc[df['sample_id'].drop_duplicates().index]
        df = df.set_index('sample_id')

        embedded_df = pd.DataFrame(df.apply(embed, axis=1))
        embedded_df.to_pickle(self.output().path)


if __name__ == '__main__':
    luigi.build([
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=100, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=50, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=200, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=300, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=100, epochs=10),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=100, epochs=15),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=100, epochs=20),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=1, size=100, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=50, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=200, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=300, epochs=5),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=100, epochs=10),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=100, epochs=15),
        EmbedBiom(aws_profile='dse', use_value=True, min_count=2, size=100, epochs=20)

    ], workers=3, local_scheduler=True)
