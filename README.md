# Big-Data-DS-1003-Final-Project-

Big Data Project - building a music recommender with ALS model.

Basic Recommender System - We have two working versions, one where we train and tune our model in EC_training.py, and load the model in EC_test.py to evaluate against test data, and another version that in EC_train_and_test.py which trains and tunes a model, and directly continues on to evaluate the recommendations against the test data.
    \begin{itemize}
        \item \textbf{Training and Tuning Model} - \texttt{spark-submit EC\_training.py root\_path\_to\_data training\_data validation\_data test\_data path\_to\_save\_model boolean\_for\_tuning}
        \item This program reads in training, validation, and test datasets in order to determine which user rows are absolutely necessary for creating the model. For the sake of testing the entire pipeline without going through hyper-parameter tuning, there is a parameter `tuning` which defaults to False. Setting it to True allows for hyper-parameter tuning.
        \item \textbf{Testing Model} - \texttt{spark-submit EC\_test.py path\_to\_test\_data  path\_to\_indexers path\_to\_load\_model}
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
