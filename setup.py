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
        'boto3',
        'pandas'
    ],
)