# Big-Data-DS-1003-Final-Project-

Big Data Project - building a music recommender with ALS model.

Basic Recommender System - We have two working versions, one where we train and tune our model in EC_training.py, and load the model in EC_test.py to evaluate against test data, and another version that in EC_train_and_test.py which trains and tunes a model, and directly continues on to evaluate the recommendations against the test data.
- Training and Tuning Model: spark-submit EC_training.py root_path_to_data training_data validation_data test_data path_to_save_model boolean_for_tuning}
- This program reads in training, validation, and test datasets in order to determine which user rows are absolutely necessary for creating the model. For the sake of testing the entire pipeline without going through hyper-parameter tuning, there is a parameter `tuning` which defaults to False. Setting it to True allows for hyper-parameter tuning.
- Testing Model: spark-submit EC_test.py path_to_test_data  path_to_indexers path_to_load_model
        \item This program loads a saved model file, a saved indexer (string indexers), and evaluates recommendations against the test data set.
        \item \textbf{Training and Testing}- \texttt{spark-submit EC\_train\_and\_test.py path\_to\_train\_data path\_to\_validation\_data path\_to\_test\_data path\_to\_model\_file boolean\_for\_tuning}
        \item This program is a complete pipeline of the previous two programs, for use when the dumbo cluster has problems with a loaded model file.
    \end{itemize}
    \item Extension 1: Alternative Model Formulations
    \begin{itemize}
        \item \textbf{Testing different count transformations} - \texttt{spark-submit EC\_training\_alt\_models.py root\_path\_to\_all\_data train\_data validation\_data test\_data model\_file1 model\_file2 model\_file3 model\_file4 boolean\_for\_tuning}
        \item This program takes in paths to all data sets, and trains a model on data after count transformations, using the hyper-parameters from our best model from the basic recommender system 
    \end{itemize}
    \item Extension 2: Fast Search
    \begin{itemize}
        \item This extension requires use of the external library \texttt{annoy}.
        \item \textbf{Testing Query Time and Precision} - \texttt{spark-submit EC\_ext2.py path\_to\_test\_file path\_to\_indexers path\_to\_model, limit}
        \item This program takes the path to test data, path to indexers (string indexers) and path to model in order to extract the necessary information of our trained brute force model. The parameter limit determines the proportions of all users we wish to query for.
    \end{itemize}
\end{itemize}
