 First run deblur_full_biom_workflow.ipynb notebook which will get the biom data from the rebiom deblur context,
remove blooms, run rarefaction, and export biom results as sparse dataframe to .pkl file. It also generates a OTU_id - dna seq
lookup dic and saves to a .csv. files written out by this notebook are saved to a 'deblur_biom_wflow_output_data' directory

Next run data_join_workflow.ipynb	 which will clean up all of the sample_ids so they are joinable for all datasets including
metadata, drug, and biom data and then it will export all of the results to 'join_wflow_output_data' dir

alpha_diversity diretory contains notbook(s) to generate alpha_diversity

bloom directory directory contains example and test of bloom filtering (which already is done in deblur_full_biom_workflow)
