# -*- coding: utf-8 -*-

# transitive closure
from american_gut_project.persist import load_dataframe, download_file, save_dataframe, upload_file
import pkg_resources
import pandas as pd
import re
import numpy as np


# download taxonomy file from S3
download_file('97_otu_taxonomy.txt', 'default')
taxonomies = load_dataframe('taxonomy.csv')

transitive_closure = []
for taxonomy in range(len(taxonomies)-1):
    hierarchy = taxonomies.iloc[taxonomy].values[0]
    # extract information
    KINGDOM = re.search('(?<=k__).*?(?=;)', hierarchy).group(0)
    PYLUM = re.search('(?<=p__).*?(?=;)', hierarchy).group(0)
    CLASS = re.search('(?<=c__).*?(?=;)', hierarchy).group(0)
    ORDER = re.search('(?<=o__).*?(?=;)', hierarchy).group(0)
    FAMILY = re.search('(?<=f__).*?(?=;)', hierarchy).group(0)
    GENUS = re.search('(?<=g__).*?(?=;)', hierarchy).group(0)
    SPECIES = re.search('(?<=^).*?(?=\t)', hierarchy).group(0)
    # start with species, then continue down branch till nothing
    ancestors = [KINGDOM, PYLUM, CLASS, ORDER, FAMILY, GENUS]
    desendents = [PYLUM, CLASS, ORDER, FAMILY, GENUS, SPECIES]
    # if descendent does not exist end with species ID
    for ancestor, descendent in zip(ancestors, desendents):
        if descendent == '':
            transitive_closure.append([ancestor, SPECIES])
            break
        else:
            transitive_closure.append([ancestor, descendent])

df = pd.DataFrame(transitive_closure, columns = ['id1', 'id2'])
df = df.drop_duplicates()
df['weight'] = 1

# save and upload file for transitive closure
save_dataframe(df, 'taxonomy_97_transitive_closure.csv')
upload_file('taxonomy_97_transitive_closure.csv', 'default')

# generate look up table for sample id to taxa
local_file_path = pkg_resources.resource_filename('american_gut_project.data',
                                                  'taxonomy.csv')

taxonomies = pd.read_csv(local_file_path, sep=';', 
                         names=['K', 'P', 'C', 'O', 'F', 'G', 'S'])

# extract greengenes species ID
get_species_id = lambda x: re.search('[0-9]*', x).group(0)
taxonomies['species_id'] = taxonomies['K'].apply(lambda x: str(get_species_id(x)))

# clean up phylum and class descriptions
taxonomies['phylum'] = taxonomies['P'].apply(lambda x: x.split('__')[1])
taxonomies['class'] = taxonomies['C'].apply(lambda x: x.split('__')[1])

taxonomies = taxonomies[['species_id', 'phylum', 'class']]

save_dataframe(taxonomies, 'speciesid_to_tax.csv')
upload_file('speciesid_to_tax.csv', 'default')
