# Big-Data-DS-1003-Final-Project-

Big Data Project - building a music recommender with ALS model.

You need Python 3.6.5 and Spark 2.4.0, and access to an HPC. 

Basic Recommender System - We have two working versions, one where we train and tune our model in `EC_training.py`, and load the model in `EC_test.py` to evaluate against test data, and another version that in `EC_train_and_test.py` which trains and tunes a model, and directly continues on to evaluate the recommendations against the test data.
- Training and Tuning Model: `spark-submit EC_training.py root_path_to_data training_data validation_data test_data path_to_save_model boolean_for_tuning`
        - This program reads in training, validation, and test datasets in order to determine which user rows are absolutely necessary for creating the model. For the sake of testing the entire pipeline without going through hyper-parameter tuning, there is a parameter `tuning` which defaults to False. Setting it to True allows for hyper-parameter tuning.
- Testing Model: `spark-submit EC_test.py path_to_test_data  path_to_indexers path_to_load_model`
        - This program loads a saved model file, a saved indexer (string indexers), and evaluates recommendations against the test data set.
- Training and Testing - `spark-submit EC_train_and_test.py path_to_train_data path_to_validation_data path_to_test_data path_to_model_file boolean_for_tuning`
        - This program is a complete pipeline of the previous two programs, for use when the dumbo cluster has problems with a loaded model file.

Extension 1: Alternative Model Formulations
- Testing different count transformations: `spark-submit EC_training_alt_models.py root_path_to_all_data train_data validation_data test_data model_file1 model_file2 model_file3 model_file4 boolean_for_tuning`
        - This program takes in paths to all data sets, and trains a model on data after count transformations, using the hyper-parameters from our best model from the basic recommender system 

Extension 2: Fast Search
- This extension requires use of the external library `annoy`.
- Testing Query Time and Precision: `ark-submit EC_ext2.py path_to_test_file path_to_indexers path_to_model, limit`
        - is program takes the path to test data, path to indexers (string indexers) and path to model in order to extract the necessary information of our trained brute force model. The parameter limit determines the proportions of all users we wish to query for.
