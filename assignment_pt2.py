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
        "Random_Hill": mlrose.random_hill_climb,
        "Simulated_Annealing": mlrose.simulated_annealing,
        "Genetic_Algorithm": mlrose.genetic_alg,
        "MIMIC": mlrose.mimic
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
        discretized_weights = np.array(weights)
        for i in range(len(weights)):
            discretized_weights[i] = min(weights_discrete_values, key=lambda x: abs(x - weights[i]))
                                        
        nn.fitted_weights = discretized_weights
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
        weights_discrete_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        problem = mlrose.DiscreteOpt(length=num_nodes, fitness_fn=mlrose.CustomFitness(discretized_fitness_function),
                             maximize=False, max_val=15)
    else:
        problem = mlrose.ContinuousOpt(num_nodes, fitness, maximize=False,
                                min_val=-1*nn.clip_max,
                                max_val=nn.clip_max, step=nn.learning_rate)
    
    #run optimization with the requested algorithm
    results = algorithms[args.algo](problem, max_attempts=10, max_iters=1000, random_state=69)
    if len(results)==2:
        best_weights, current_loss = results
    else:
        best_weights, current_loss, _ = results

    #evaluate
    nn.fitted_weights = best_weights
    nn.output_activation = fitness.get_output_activation()
    y_test_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy", test_acc)

if __name__=='__main__':
    main()