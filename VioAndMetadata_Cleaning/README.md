## Pure Metadata Survey Analysis 

#### Summary

This area of the project contains notebooks that process a sample of the qiita download of AGP metadata and does analysis on it,
preprocessing, imputation, and eventually ML outputs as a trial/experiment.

1. The first notebook to be run is CleanUpMetadataNoVioUse.ipynb, which will clean the metadata and put into numerical categories
2. Next, since the previous step outputted cleanedUpMetadata_noVio_AGP_humfece.csv, the next notebook, metadataImputationAnalysisAGP.ipynb
will take that .csv in, and run imputation techniques to try and fill in missing data, as well as showing a trial to see MSE results on 40 samples
3. The last notebook, machineLearningSurveyTests.ipynb will take just metadata survey after the cleanup and imputation, and will
run random forests on it to try and predict various target variables and output results.