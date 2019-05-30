# -*- coding: utf-8 -*-

# Body Site Classification
from american_gut_project.persist import load_dataframe, download_file, save_dataframe, upload_file
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def split_sampleid(global_id):
    sample_id = global_id.split('.')[1]
    try:
        return str(int(sample_id))
    except ValueError:
        return None


# download data from S3
download_file('deblur_all_body_4.16.rar1000_clean_biom.pkl', profile_name='default')
samples = load_dataframe('4.10.rar1000.biom_data.pkl')
samples = samples.fillna(0)

# set sample name as index and drop it as a column
samples.index = samples['sample_name']
samples = samples.drop(columns='sample_name')

# load embeddings
download_file('taxonomy_97_lorentz_embedding.csv', profile_name='default')
df_embeddings = pd.read_csv(r'C:\Users\bwesterber\Downloads\taxonomy_97_lorentz_embedding.csv', index_col=0)

# load species ID to taxonomy mapping for visualization
download_file('speciesid_to_tax.csv', 'default')
taxonomies = load_dataframe('speciesid_to_tax.csv')
taxonomies['species_id'] = taxonomies['species_id'].apply(lambda x: str(x))

# apply embeddings
embeddings = []
sample_indicies = []
sample_ids = samples.index.tolist()
unique_otus = samples.columns
otu_tax = []

for otu in range(len(samples)):
    print(otu)
    # select otus where count is > 0 and log scale
    otu_weights = pd.DataFrame(np.log(samples.iloc[otu][samples.iloc[otu] > 0] + 1))
    # join otu counts, species id
    merged_otu_tax = otu_weights.merge(taxonomies, how='left', left_index=True,
                                       right_on='species_id')
    # drop nans
    merged_otu_tax = merged_otu_tax.dropna(axis=0)
    merged_otu_tax['sample_id'] = otu_weights.columns[0]

    merged_otu_tax.columns = ['weight', 'species_id', 'phylum', 'class',
                              'sample_id']
    # append to list for concatenation
    otu_tax.append(merged_otu_tax)
    # cast to string to match other index
    otu_weights.index = [str(x) for x in otu_weights.index]
    otu_weights.columns = ['weight']
    # embeddings for otus in this sample, inner join to deal with out of vocabulary otus
    sample_embeddings = df_embeddings.merge(otu_weights, how='inner',
                                            left_index=True, right_index=True)
    # multiply each otu vector by its weight, split the joined columnss
    otu_weights = sample_embeddings['weight']
    sample_embeddings = sample_embeddings.drop(columns='weight')
    # calculate average
    sample_embedding = np.sum(sample_embeddings.multiply(otu_weights, axis=0))/len(sample_embeddings)
    embeddings.append(np.array(sample_embedding))

df_embedded = pd.DataFrame(embeddings, index=sample_ids)
df_embedded['global_sample_id'] = sample_ids

# convert global names to sample ids for American Gut Project
df_embedded['sample_id'] = df_embedded['global_sample_id'].apply(lambda x: split_sampleid(x))
df_embedded.index = df_embedded['sample_id']
df_embedded = df_embedded.drop(columns=['global_sample_id', 'sample_id'])

# concatenate and clean up visualization data otu --> taxonomy
df_otu_tax = pd.concat(otu_tax, axis=0)
df_otu_tax['sample_id'] = df_otu_tax['sample_id'].apply(lambda x: split_sampleid(x))

# save and upload sample id to taxonomy mapping
save_dataframe(df_otu_tax, 'sampleid_to_tax.csv')
upload_file('sampleid_to_tax.csv', 'default')

'''
 AGP Metadata:
 Join embeddings with body site data for classification
 '''
download_file('agp_only_meta.csv', profile_name='default')
df_meta = load_dataframe('agp_only_meta.csv')
df_meta = df_meta[['anonymized_name', 'env_material']]
df_meta = df_meta.dropna(axis=0)

df_meta.index = df_meta['anonymized_name'].apply(lambda x: str(int(x)))
df_meta = df_meta.drop(columns='anonymized_name')


# inner join on embeddings
df_embedded = df_embedded.merge(df_meta, how='inner', left_index=True, right_index=True)
label = df_embedded['env_material']
df_embedded = df_embedded.drop(columns='env_material')


# body site classification
x_train, x_test, y_train, y_test = train_test_split(df_embedded, label, test_size=0.33, random_state=0)

# classification
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

# train
print(classification_report(y_pred_train, y_train))
print(accuracy_score(y_pred_train, y_train))

# test
print(classification_report(y_pred_test, y_test))
print(accuracy_score(y_pred_test, y_test))

# visualization
pca = PCA(n_components=3, svd_solver='full')
pca.fit(df_embedded)
df_pca = pd.DataFrame(pca.transform(df_embedded))
df_pca.index = np.array(df_embedded.index)

# add label
df_pca['label'] = label

colors = ['#00C6D7', '#6E963B', '#FC8900']
labels = ['feces', 'saliva', 'sebum']

fig = plt.figure(figsize=(8,5))

for color, label in zip(colors, labels):
    df_label = df_pca[df_pca['label'] == label]
    plt.scatter(df_label[0], df_label[1], alpha=0.5, color=color)

plt.title('Hyperbolic Microbiome Embedding')
plt.xlabel('PC1 ({}%)'.format(np.round(pca.explained_variance_ratio_[0]*100, 2)))
plt.ylabel('PC2 ({}%)'.format(np.round(pca.explained_variance_ratio_[1]*100, 2)))

patch_1 = mpatches.Patch(color='#00C6D7', label='Feces')
patch_2 = mpatches.Patch(color='#6E963B', label='Saliva')
patch_3 = mpatches.Patch(color='#FC8900', label='Sebum')
plt.legend(handles=[patch_1, patch_2, patch_3], loc = 1)
