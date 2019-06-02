

run all_body_types_data_join_workflow.ipynb	 which will clean up all of the sample_ids so they are joinable for all datasets including metadata, drug, and biom data and then it will export all of the results to 'join_wflow_output_data' dir.

Note: the metadata, drug data, and biom .pkl files are read in from the 'data' directory. 

The resulting data files in 'join_wflow_output_data' will all be joinable on sample_id and ready for analysis
