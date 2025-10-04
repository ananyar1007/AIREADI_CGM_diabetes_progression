# To reproduce the results for the paper,
## Run: CGM_model_training.ipynb to train all models that run on CGM stats. The models are
 Random Forest on the 6 hand crafted features
 Deep learning CNN model trained on one week of CGM data
 Deep learning CNN model trained on one day of CGM data, test on each day and take max prob
 Transformer model using CNN model to create embeddings for each day with a transformer encoder to combine the information
 XGBoost with PCA on the time series data
 TabPFN with PCA on time series data

 ## Run HbA1c_model_training.ipynb to run a random forest on HbA1c features


## If you want to extract the data from AIREADI dataset

 To create the pkl and csv files containing relevant data extracted from the AI readi dataset run:
 create_df_cgm_data.ipynb => Create the pkl file: dataframe_with_glucose_info.pkl
 To create the HbA1c data csv file, run:
 create_HbA1c_data.ipynb => Creates the file HbA1c_all_patients.csv

 Note that to be able to run these files, you need the AI_READI dataset downloaded.