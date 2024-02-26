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
        "rhc": (mlrose.RHCRunner, {'experiment_name':'tst', 'seed': 69, 'iteration_list':[100, 500, 1000], 'restart_list':[0, 10, 50]}),
        "sa": [mlrose.SARunner, {'experiment_name':'tst', 'seed': 69, 'iteration_list':[100, 500, 1000], 'max_attempts':1000, 'temperature_list':[1e10, 1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]}],
        "ga": [mlrose.GARunner, {'population_sizes': [100, 200], 'mutation_rates':[0.1, 0.3, 0.2], 'iteration_list':[100,300,500, 1000], 'seed': 69, 'experiment_name': 'tst'}],
        "mimic": [mlrose.MIMICRunner, {'population_sizes': [200, 200], 'keep_percent_list':[0.1, 0.15, 0.5,0.65], 'iteration_list':[100,300,500, 500, 1000], 'seed': 69, 'experiment_name': 'tst', 'use_fast_mimic':True}]
    }

    #dataset prep
    X_train, X_test, y_train, y_test = genData(args.dataset)
    #create the NN
    nn = mlrose.NeuralNetwork(hidden_nodes=[50,70], activation='relu',
                           algorithm='gradient_descent', max_iters=1000,
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=True, clip_max=5, max_attempts=100,
                           random_state=42)
    input_nodes = np.shape(X_train)[1] + nn.bias
    if len(np.shape(y_train))==2:
        output_nodes = np.shape(y_train)[1]
    else:
        output_nodes = 1
    node_list = [input_nodes] + nn.hidden_nodes + [output_nodes]
    nn.node_list = node_list

    # Define the discretized fitness function
    def discretized_fitness_function(weights):
        # Discretize the weights
        normalized_weights = np.array(weights)
        min_weight = -0.5
        max_weight = 0.5

        normalized_weights = min_weight + (max_weight - min_weight) * (normalized_weights / 15)
                                        
        nn.fitted_weights = normalized_weights
        nn.node_list = node_list
        nn.output_activation = mlrose.sigmoid
    
        # Evaluate fitness based on performance
        fitness = nn.score(X_train, y_train)  # Adjust as needed based on available data
        
        return fitness
    #---------------end fitness function------------------------
    

    num_nodes = 0

    for i in range(len(node_list) - 1):
            num_nodes += node_list[i]*node_list[i+1]

    #define the optimization problem
    fitness = mlrose.NetworkWeights(X_train, y_train, node_list,
                                 nn.activation_dict[nn.activation],
                                 nn.bias, nn.is_classifier,
                                 learning_rate=nn.learning_rate)
    if args.algo=='MIMIC':
        problem = mlrose.DiscreteOpt(length=num_nodes, fitness_fn=mlrose.CustomFitness(discretized_fitness_function),
                             maximize=False, max_val=15)
    else:
        problem = mlrose.ContinuousOpt(num_nodes, fitness, maximize=False,
                                min_val=-1*nn.clip_max,
                                max_val=nn.clip_max, step=nn.learning_rate)
    
    #run optimization with the requested algorithm
    results = algorithms[args.algo][0](problem, **algorithms[args.algo][1])
    if len(results)==2:
        best_weights, current_loss = results
    else:
        best_weights, current_loss, _ = results

    #evaluate
    if args.algo == 'MIMIC':
        min_weight = -0.5
        max_weight = 0.5

        best_weights = min_weight + (max_weight - min_weight) * (best_weights / 15)
    nn.fitted_weights = best_weights
    nn.output_activation = fitness.get_output_activation()
    y_test_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy", test_acc)

if __name__=='__main__':
    main()