from american_gut_project.persist import load_dataframe


def process_metadata():
    df = load_dataframe('AGP_metdata.csv')

    print(df)


def process_drug_data():
    df = load_dataframe('drug_data_dense.csv')

    print(df)


def process_biome_data():
    df = load_dataframe('biom_table.pkl')

    print(df)

if __name__ == '__main__':
    process_biome_data()
