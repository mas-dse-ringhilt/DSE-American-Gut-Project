from random import shuffle

import gensim
import pandas as pd

from american_gut_project.persist import save_w2v_model, load_dataframe


def build_microbiome_embeddings():
    # pull table from local file
    samples = load_dataframe('biom_table.pkl')

    # replace nans with zeros
    samples = samples.fillna(0)

    sentences = []
    for sample in range(len(samples) - 1):
        try:
            sentence = list(samples[sample][samples[sample] > 0].index)
            sentences.append([str(x) for x in sentence])
        except KeyError:
            pass

    # shuffle each sample around and append it to the training data
    augmentation_constant = 2
    generated_sentences = []
    for sentence in sentences:
        for augmentation in range(augmentation_constant):
            shuffle(sentence)
            generated_sentences.append(sentence)

    model = gensim.models.Word2Vec(generated_sentences, size=100, min_count=2, window=100, workers=4, sg=0)
    model.train(generated_sentences, total_examples=len(sentences), epochs=5)
    save_w2v_model(model, 'microbiome_w2v.model')

if __name__ == '__main__':
    build_microbiome_embeddings()
