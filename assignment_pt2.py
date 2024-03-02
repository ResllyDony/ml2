from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import random
import argparse

def genData(datadir, target_col=-1, test_size=0.2, random_state=69, split=True):
    def quantize(column):
        if column.dtype == 'object':
            return column.astype('category').cat.codes
        return column
    df = pd.read_csv(datadir, index_col=None)
    df = df.dropna()
    df = df.apply(quantize)
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable
    if not split:
        return X, y
    # df=df.drop(df.columns[drop_cols], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def main():
    np.random.seed(69)

    #input handler
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, help='Dataset to train the NN model on.')
    parser.add_argument('-a', '--algo', required=True, help='Random opt algorithm to train the model using.')
    args = parser.parse_args()

    # Define optimization algorithms
    algorithms = {
        "rhc": (mlrose.random_hill_climb, {'seed': [69], 'iteration_list':[100, 500, 1000], 'restart_list':[0, 10, 50], 'max_iters': [1, 2, 4, 8, 16, 32, 64, 128], 'learning_rate': [0.001, 0.002, 0.003]}),
        "sa": [mlrose.simulated_annealing, {'seed': [69], 'iteration_list':[100, 500, 1000], 'max_attempts':[1, 1000], 'temperature_list':[1e10, 1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000], 'max_iters': [1, 2, 4, 8, 16, 32, 64, 128], 'learning_rate': [0.001, 0.002, 0.003]}],
        "ga": [mlrose.genetic_alg, {'population_sizes': [100, 200, 500], 'mutation_rates':[0.1, 0.3, 0.2], 'iteration_list':[100,300,500, 1000], 'seed': [69],  'max_iters': [1, 2, 4, 8, 16, 32, 64, 128], 'learning_rate': [0.001, 0.002, 0.003]}],
        "mimic": [mlrose.mimic, {'population_sizes': [200, 200], 'keep_percent_list':[0.1, 0.15, 0.5,0.65], 'iteration_list':[100,300,500, 500, 1000], 'seed': [69], 'use_fast_mimic':True, 'max_iters': [1, 2, 4, 8, 16, 32, 64, 128], 'learning_rate': [0.001, 0.002, 0.003]}]
    }

    #dataset prep
    X_train, X_test, y_train, y_test = genData(args.dataset)
    nnr = mlrose.NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        experiment_name='nn_test',
        algorithm=algorithms[args.algo][0],
        grid_search_parameters=algorithms[args.algo][1],
        iteration_list=[1, 10, 50, 100, 250, 500, 1000],
        hidden_layer_sizes=[[10]],
        bias=True,
        early_stopping=False,
        clip_max=1e+10,
        max_attempts=500,
        generate_curves=True,
        seed=69
    )
    results = nnr.run()
    print(results)

if __name__=='__main__':
    main()