from setuptools import setup, find_packages

setup(
    # Package information
    name='american_gut_project',
    version='0.0.1',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'awscli',
        'boto3',
        'pandas',
        'gensim',
        'sklearn',
        'tables',
        'luigi',
        'xgboost',
        'numpy',
    ],

    entry_points={
        'console_scripts': [
            'agp-pipeline=american_gut_project.main:arg_parse_pipeline',
            'agp-push=american_gut_project.main:arg_parse_push',
            'agp-pull=american_gut_project.main:arg_parse_pull',
        ],
    }

)
