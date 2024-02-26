import mlrose_hiive as mlrose
import numpy as np
import networkx as nx
import random
import argparse

def experiment1():
    '''Traveling Salesman Problem: Minimize'''
    # Create a random graph with 100 nodes
    G = nx.random_geometric_graph(50, 0.5)
    # Extract node coordinates
    coords_list = [tuple(map(float, coord)) for coord in nx.get_node_attributes(G, 'pos').values()]
    # Define optimization problem object
    problem = mlrose.TSPOpt(length=len(coords_list), coords=coords_list, maximize=False)
    return problem

def experiment2():
    '''Flip Flop problem: Maximize'''
    
    # Define optimization problem object
    problem = mlrose.FlipFlopOpt(length=1000, maximize=True)
    return problem

def experiment3():
    '''Four peaks problem: Maximize'''
    # Define fitness function
    fitness = mlrose.FourPeaks(t_pct=0.15)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=200, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment4():
    '''six peaks problem: Maximize'''
    # Define fitness function
    fitness = mlrose.SixPeaks(t_pct=0.15)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=200, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment5():
    '''Continuous peaks problem: Maximize'''
    # Define fitness function
    fitness = mlrose.ContinuousPeaks(t_pct=0.15)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=200, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment6():
    '''Knapsack Problem: Maximize'''
    # Define random weights and values for items
    num_items = 20
    weights = np.random.randint(1, 20, size=num_items)
    values = np.random.randint(1, 20, size=num_items)
    
    # Define maximum weight capacity of the knapsack
    max_weight_pct = 0.5  # Maximum weight capacity as a percentage of total weight
    
    # Define fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    
    # Define optimization problem object
    problem = mlrose.KnapsackOpt(length=num_items, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment7():
    '''N-Queens Problem: Minimize'''
    # Define the size of the board (number of queens)
    board_size = 8  # Increase the size of the chessboard
    
    # Define optimization problem object
    problem = mlrose.QueensOpt(length=board_size, fitness_fn=fitness, maximize=False)
    return problem

def experiment8():
    '''Max K-Color problem: Mimimize'''
    G = nx.gnp_random_graph(n=20, p=0.5)
    edges = list(G.edges())
    
    # Define optimization problem object
    problem = mlrose.MaxKColorOpt(edges=edges, maximize=False, max_colors=5)
    return problem



def main():
    np.random.seed(69)
    random.seed(69)

    #arg handler
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', required=True, help='select experiment # (1-3). Runs all experiments by default')
    parser.add_argument('-a', '--algo', required=True, help='select the opt alrogirhm. Runs all by default')
    args = parser.parse_args()

    # Define optimization algorithms
    algorithms = {
        "rhc": (mlrose.RHCRunner, {'experiment_name':'tst', 'seed': 69, 'iteration_list':[100,300,500], 'restart_list':[20,45,70]}),
        "sa": [mlrose.SARunner, {'experiment_name':'tst', 'seed': 69, 'iteration_list':[100,300,500], 'max_attempts':1000, 'temperature_list':[10, 50, 100]}],
        "ga": [mlrose.GARunner, {'population_sizes': [20,50,100, 500], 'mutation_rates':[0.1, 0.1, 0.2], 'iteration_list':[100,300,500], 'seed': 69, 'experiment_name': 'tst'}],
        "mimic": [mlrose.MIMICRunner, {'population_sizes': [20,50,100, 200, 500], 'keep_percent_list':[0.1, 0.15, 0.5,0.65], 'iteration_list':[100,300,500, 500], 'seed': 69, 'experiment_name': 'tst', 'use_fast_mimic':True}]
    }

    # Loop through problems and algorithms
    # problems = [experiment1, experiment2, experiment3, experiment4, experiment5]
    problems = [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7 ,experiment8]
    problem = problems[int(args.experiment)-1]()
    print(f"Problem {args.experiment}: {problem.__class__.__name__}")
    

    print(f"\tAlgorithm: {args.algo}")
    alg_func=algorithms[args.algo]
    # Run algorithm
    result = alg_func[0](problem, **alg_func[1]).run()

    if isinstance(result, tuple) and len(result) == 3:
        best_state, best_fitness, _ = result
        print(f"\t\tBest Fitness: {best_fitness}")
    if isinstance(result, tuple) and len(result) == 2:
        best_state, best_fitness = result
        print(f"\t\tBest Fitness: \n{best_fitness}")
        print(f'\t\trun stats:\n{best_state}')
    
    # Output results
    # print(f"\t\tBest Fitness: {best_fitness}")

if __name__=='__main__':
    main()