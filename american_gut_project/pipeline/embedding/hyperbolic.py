# -*- coding: utf-8 -*-

# Body Site Classification
from american_gut_project.persist import load_dataframe, download_file, save_dataframe, upload_file
import pandas as pd
import numpy as np
import luigi
from american_gut_project.pipeline.fetch import FetchData
from american_gut_project.paths import paths
import re


def split_sampleid(global_id):
    sample_id = global_id.split('.')[1]
    try:
        return str(int(sample_id))
    except ValueError:
        return None


def get_species_id(x):
    return re.search('[0-9]*', x).group(0)


class HyperbolicPreproccessing(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        output_paths = [
            'speciesid_to_tax.csv',
            'taxonomy_97_transitive_closure.csv'
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [
            FetchData(filename='taxonomy_97_lorentz_embedding.csv', aws_profile=self.aws_profile),
            FetchData(filename='taxonomy.csv', aws_profile=self.aws_profile),

        ]

    def run(self):
        taxonomies = pd.read_csv(self.input()[1].path, index_col=0)

        transitive_closure = []
        for taxonomy in range(len(taxonomies) - 1):
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

        df = pd.DataFrame(transitive_closure, columns=['id1', 'id2'])
        df = df.drop_duplicates()
        df['weight'] = 1

        # # save and upload file for transitive closure
        # save_dataframe(df, 'taxonomy_97_transitive_closure.csv')
        # upload_file('taxonomy_97_transitive_closure.csv', 'default')
        output_path = self.output()[1].path
        df.to_csv(output_path)

        taxonomies = pd.read_csv(self.input()[1].path, sep=';',
                                 names=['K', 'P', 'C', 'O', 'F', 'G', 'S'])

        # extract greengenes species ID
        get_species_id = lambda x: re.search('[0-9]*', x).group(0)
        taxonomies['species_id'] = taxonomies['K'].apply(lambda x: str(get_species_id(x)))

        # clean up phylum and class descriptions
        taxonomies['phylum'] = taxonomies['P'].apply(lambda x: x.split('__')[1])
        taxonomies['class'] = taxonomies['C'].apply(lambda x: x.split('__')[1])

        taxonomies = taxonomies[['species_id', 'phylum', 'class']]

        output_path = self.output()[0].path
        taxonomies.to_csv(output_path)


class Hyperbolic(luigi.Task):
    aws_profile = luigi.Parameter(default='default')

    def output(self):
        output_paths = [
            'sample_id_to_tax.csv',
            'hyperbolic_df.pkl'
        ]

        outputs = [paths.output(p) for p in output_paths]
        return [luigi.LocalTarget(output) for output in outputs]

    def requires(self):
        return [
            FetchData(filename='4.10.rar1000.biom_data.pkl', aws_profile=self.aws_profile),
            FetchData(filename='taxonomy_97_lorentz_embedding.csv', aws_profile=self.aws_profile),
            HyperbolicPreproccessing(aws_profile=self.aws_profile),
            FetchData(filename='agp_only_meta.csv', aws_profile=self.aws_profile)
        ]

    def run(self):
        samples = pd.read_pickle(self.input()[0].fn).to_dense()
        samples = samples.fillna(0)

        samples = samples.loc[samples['sample_name'].drop_duplicates().index]

        # set sample name as index and drop it as a column
        samples.index = samples['sample_name']
        samples = samples.drop(columns='sample_name')

        df_embeddings = pd.read_csv(self.input()[1].fn, index_col=0)

        # load species ID to taxonomy mapping for visualization
        taxonomies = pd.read_csv(self.input()[2][0].fn, index_col=0)
        taxonomies['species_id'] = taxonomies['species_id'].apply(lambda x: str(x))

        # apply embeddings
        embeddings = []
        sample_ids = samples.index.tolist()
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
        df_embedded = df_embedded.loc[df_embedded['sample_id'].drop_duplicates().index]
        df_embedded.index = df_embedded['sample_id']
        df_embedded = df_embedded.drop(columns=['global_sample_id', 'sample_id'])

        # concatenate and clean up visualization data otu --> taxonomy
        df_otu_tax = pd.concat(otu_tax, axis=0)
        df_otu_tax['sample_id'] = df_otu_tax['sample_id'].apply(lambda x: split_sampleid(x))

        # save and upload sample id to taxonomy mapping
        sample_id_to_tax_path = self.output()[0].path
        df_otu_tax.to_csv(sample_id_to_tax_path)

        output_path = self.output()[1].path
        df_embedded.to_pickle(output_path)


if __name__ == '__main__':
    luigi.build([Hyperbolic(aws_profile='dse')], workers=1, local_scheduler=True)


