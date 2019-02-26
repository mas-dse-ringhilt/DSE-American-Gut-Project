from american_gut_project.persist import download_file


def fetch_metadata(profile_name='default'):
    download_file('agp_only_meta.csv', profile_name)


def fetch_drug_data(profile_name='default'):
    download_file('drug_data_dense.csv', profile_name)


def fetch_biome_data(profile_name='default'):
    download_file('biom_table.pkl', profile_name)


def fetch_all(profile_name='default'):
    fetch_metadata(profile_name)
    fetch_drug_data(profile_name)
    fetch_biome_data(profile_name)


if __name__ == '__main__':
    fetch_all('dse')
