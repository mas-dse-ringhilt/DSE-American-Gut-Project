

run all_body_types_data_join_workflow.ipynb	 which will clean up all of the sample_ids so they are joinable for all datasets including metadata, drug, and biom data and then it will export all of the results to 'join_wflow_output_data' dir.

The metadata file referenced (raw.2.21.agp_metadata.txt) comes from https://qiita.ucsd.edu/study/description/10317, select 'sample information' and then 'download sample info which will download the latest metadata results.

Instead of downloading the metadata file manually from the site above, you can also pull the metadata information using the redbiom api (see the AG_Example notebook under 'Notebooks/redbiom/AG_example.ipynb'

Note: the metadata, drug data, and biom .pkl files are read in from the 'data' directory. 

The resulting data files in 'join_wflow_output_data' will all be joinable on sample_id and ready for analysis
