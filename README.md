# DSE-American-Gut-Project

## Repository Info

### Data Cleaning / Integration
The 'data_join_workflow' directory contains a jupyter notebook that pulls in data from all 3 data sources (raw metadata .txt file, biom .pkl file, and drug questionnaire .csv). The notebook cleans the data, including the extraction / creation of a consistent 'sample_id' which is joinable across the 3 datasets. The notebook outputs cleaned and joinable versions of the different data sources, ready for analysis.

Please see the README.md in the 'data_join_workflow' directory for more information

### Pipeline

The pipeline is under the american_gut_project folder, and creates a library as such.

Simple steps to install:

1. From the root of the git repo, run `pip install -e .` which installs the packages needed and the newly made library
2. Run a test python file, like dataset.py, which preprocesses data
3. Run one of the larger pipelines, like under american_gut_project/pipeline/model (american_gut_project.pipeline.model), to run a simple_model.py or others

## References

Consult the QIIME2 and Qiita documentation that can be found online here:
https://qiime2.org/
https://qiita.ucsd.edu/

Metagenomics Info: http://www.metagenomics.wiki/pdf/definition

BIOM format: http://biom-format.org/

#### Papers

American Gut: https://msystems.asm.org/content/3/3/e00031-18

Conducting a Microbiome Study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5074386/

A Guide to Enterotypes across the Human Body: Meta-Analysis of Microbial Community Structures in Human Microbiome Datasets:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3542080/

Supervised classification of human microbiota: https://www.ncbi.nlm.nih.gov/pubmed/21039646
