# ProjetTC1
## Challenge Otto Kaggle

Disponible à https://www.kaggle.com/c/otto-group-product-classification-challenge


## How to use
Two files can be run : 

### algo_tuning.py 
It run a naive grid search for many parameters of 5 algorithms : 

    Random Forest
    XGBoost
    Mutli-Layer Perceptron
    AdaBoost
    Extra Trees

It also plots logloss computed on validation set according to each parameter's value.

You just have to change the line `X, y, test_X = dex.get_data("../train.csv", "../test.csv")`
    
where `dex.get_data` function needs to have the correct path towards train and test csv files.


### algorithms.py 
It has two main functions : 

main_test aims to train (on train set)

    Random Forest
    XGBoost
    Mutli-Layer Perceptron
    Extra Trees
    
and write csv submission files and csv validation probability files (for faster computation), after which it tries random weights to combine multiple probabilities.

Those weights can be used in the second function to write final csv submission files.

main_submit which train the same classifiers on the full dataset. It then compute combined probabilities than are recorded in final csv submission files. The weigths for combining are to be given manually.

Same thing as before, you just have to change the line `X, y, test_X = dex.get_data("../train.csv", "../test.csv")`
    
where `dex.get_data` function needs to have the correct path towards train and test csv files.

### Run
To run either one, `python algo_tuning.py` or `python algorithms.py`
    
