from random import shuffle

import gensim
import luigi
import pandas as pd
import itertools
import numpy as np
from gensim.models.poincare import PoincareModel

from american_gut_project_pipeline.persist import save_w2v_model, load_dataframe


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
    

def otu_transitive_closure(df):
    """ Build the transitive closure for the OTU coocurrence graph. If OTUs
        cooccur in the same microbiome sample they share an edge with each
        other. Assumes each microbiome sample has its own colum and each 
        OTU is a row. 
    """
    edges = []
    columns = df.columns
    for sample in range(len(df)):
        mask = df.iloc[sample] > 0
        verticies =  list(np.array(mask)*np.array(columns))
        verticies = [str(x) for x in verticies if x != '']
        if len(verticies) > 1:
            edges.append(list(itertools.permutations(verticies, 2)))
    return set([x for sample in edges for x in sample])


def build_poincare_embedding(graph):
    poincare_model = PoincareModel(graph, burn_in=10, negative=2, alpha=1)
    poincare_model.train(epochs=1)
    save_w2v_model(poincare_model, 'microbiome_poincare.model')


if __name__ == '__main__':
    build_microbiome_embeddings()
