# DSE-American-Gut-Project

## Repository Info

### Biom Data Ingestion
The 'biom_data_ingest' directory contains notebooks which pull the American Gut Project biom OTU data of interest from the Redbiom python API, removes bloom OTU's from the biom table data, runs rarefaction on the data, and then finally exports it into a sparse dataframe (in .pkl format).

Two different ingestion notebooks exist within the directory, one pulls from 'greengenes' context, the other 'deblur' context. See the notebooks for more detailed information on the ingestion process.

You can also look at 'Notebooks/redbiom/AG_example.ipynb' for a more detailed walkthrough example of the biom ingestion process as well'

### Alpha Diversity
The 'biom_data_ingest' directory also contains 'greengenes_alpha_diversity.ipynb' which calculates phylogenic alpha diversity of the biom data samples, using the greengenes 97 tree. 

### Beta Diversity / pcOa
The 'beta_diversity_pcOa' directory contains notebooks which calculate phylogenic beta diversity between samples using the greengenes 97 tree, as well as running principal coordinates analysis on the beta diversity matrix results.

### Data Cleaning / Integration
The 'data_join_workflow' directory contains a jupyter notebook that pulls in data from all 3 data sources (raw metadata .txt file, biom .pkl file, and drug questionnaire .csv). The notebook cleans the data, including the extraction / creation of a consistent 'sample_id' which is joinable across the 3 datasets. The notebook outputs cleaned and joinable versions of the different data sources, ready for analysis.

Please see the README.md in the 'data_join_workflow' directory for more information

The cleaned output data sources are used as the input data into our pipeline. The pipeline is used for multiple analyitical workflows including preprocessing, hyperbolic and word2vec embeddings, hyperparamter tuning, and supervised machine learning classifcation (See Pipeline section below)

### Pipeline

The pipeline is under the american_gut_project_pipeline folder, and creates a library as such.

#### Installation
1. Acquire AWS Credentials to pull files from S3. Make sure they are saved into `~/.aws/credentials` file
2. From the root of the git repo, run `pip install -e .` which installs the packages needed and the newly made library

#### Executing
 The entire pipeline can be executed by running `agp-pipeline` in the terminal. If the AWS credentials are not in the
`[default]` block within the credential file, the command can point to another credential like this `agp-pipeline -a 
<aws_profile_name>`

Alternatively, all the output and intermediate data artifacts are stored within S3. Running `agp-pull` (or with `-a`) 
will sync the local file system with the data artifacts on S3. There is also an `agp-push` command that uploads the 
files to S3. 

The data artifacts from the pipeline are stored under the `data` directory

It is also possible to run single transformations (or tasks) by editing the main function in an individual file and 
executing it from the command line or through an IDE. Note that since luigi is handling the dependency management of 
the artifacts, if the output artifacts of a task exist in the `data` directory, luigi won't recompute the task unless 
the file is moved or deleted.


### Pure Metadata Survey Analysis

The directory `VioAndMetadata_Cleaning` contains analysis on purely metadata survey information and no microbiome to investigate 
what information might already be achieved with surveys and data science.  Please read the README and investigate notebooks for
more information.

Imputation techniques with SoftImpute, KNN, and Iterative Imputer along with tests are located within this directory

## References

Consult the QIIME2 and Qiita documentation that can be found online here:
https://qiime2.org/
https://qiita.ucsd.edu/ (AGP here: https://qiita.ucsd.edu/study/description/10317)

Metagenomics Info: http://www.metagenomics.wiki/pdf/definition

BIOM format: http://biom-format.org/

AGP notebooks: https://github.com/knightlab-analyses/american-gut-analyses/tree/master/ipynb

#### Papers

American Gut: https://msystems.asm.org/content/3/3/e00031-18

Conducting a Microbiome Study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5074386/

A Guide to Enterotypes across the Human Body: Meta-Analysis of Microbial Community Structures in Human Microbiome Datasets:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3542080/

Supervised classification of human microbiota: https://www.ncbi.nlm.nih.gov/pubmed/21039646
