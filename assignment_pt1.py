import mlrose_hiive as mlrose
import numpy as np
import networkx as nx
import random
import argparse
import json

# def experiment1():
#     '''Traveling Salesman Problem: Minimize'''
#     # Create a random graph with 100 nodes
#     G = nx.random_geometric_graph(30, 0.5)
#     # Extract node coordinates
#     coords_list = [tuple(map(float, coord)) for coord in nx.get_node_attributes(G, 'pos').values()]
#     # Define optimization problem object
#     problem = mlrose.TSPOpt(length=len(coords_list), coords=coords_list, maximize=False)
#     return problem, 'minimize'

def experiment1():
    '''Flip Flop problem: Maximize'''
    
    # Define optimization problem object
    problem = mlrose.FlipFlopOpt(length=500, maximize=True)
    return problem, 'maximize'

def experiment2():
    '''Four peaks problem: Maximize'''
    # Define fitness function
    fitness = mlrose.FourPeaks(t_pct=0.25)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=500, fitness_fn=fitness, maximize=True, max_val=2)
    return problem, 'maximize'

# def experiment4():
#     '''six peaks problem: Maximize'''
#     # Define fitness function
#     fitness = mlrose.SixPeaks(t_pct=0.15)
#     # Define optimization problem object
#     problem = mlrose.DiscreteOpt(length=300, fitness_fn=fitness, maximize=True, max_val=2)
#     return problem, 'maximize'

# def experiment5():
#     '''Continuous peaks problem: Maximize'''
#     # Define fitness function
#     fitness = mlrose.ContinuousPeaks(t_pct=0.15)
#     # Define optimization problem object
#     problem = mlrose.DiscreteOpt(length=300, fitness_fn=fitness, maximize=True, max_val=2)
#     return problem, 'maximize'

def experiment3():
    '''Knapsack Problem: Maximize'''
    # Define random weights and values for items
    num_items = 500
    weights = np.random.randint(1, 100, size=num_items)
    values = np.random.randint(1, 200, size=num_items)
    
    # Define maximum weight capacity of the knapsack
    max_weight_pct = 0.7  # Maximum weight capacity as a percentage of total weight
    
    # Define fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    
    # Define optimization problem object
    problem = mlrose.KnapsackOpt(length=num_items, fitness_fn=fitness, maximize=True, max_val=2)
    return problem, 'maximize'

# def experiment7():
#     '''N-Queens Problem: Minimize'''
#     # Define the size of the board (number of queens)
#     board_size = 30  # Increase the size of the chessboard
    
#     # Define optimization problem object
#     problem = mlrose.QueensOpt(length=board_size, maximize=False)
#     return problem, 'minimize'

# def experiment8():
#     '''Max K-Color problem: Mimimize'''
#     G = nx.gnp_random_graph(n=25, p=0.5)
#     edges = list(G.edges())
    
#     # Define optimization problem object
#     problem = mlrose.MaxKColorOpt(edges=edges, maximize=False, max_colors=5)
#     return problem, 'minimize'



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
        "rhc": mlrose.RHCRunner,
        "sa": mlrose.SARunner,
        "ga": mlrose.GARunner,
        "mimic": mlrose.MIMICRunner,
    }

    with open('./presets.json', 'r') as f:
        param_def = json.load(f)
    params = param_def[args.algo][f'experiment{args.experiment}']

    # Loop through problems and algorithms
    # problems = [experiment1, experiment2, experiment3, experiment4, experiment5]
    problems = [experiment1, experiment2, experiment3]
    problem, opt = problems[int(args.experiment)-1]()
    print(f"Problem {args.experiment}: {problem.__class__.__name__}")
    

    print(f"\tAlgorithm: {args.algo}")
    alg_func=algorithms[args.algo]
    # Run algorithm
    result = alg_func(problem, generate_curves=True, **params).run()

    if isinstance(result, tuple) and len(result) == 3:
        best_state, best_fitness, _ = result

    elif isinstance(result, tuple) and len(result) == 2:
        best_state, best_fitness = result

    if args.algo=='rhc':
        tmp = best_fitness[best_fitness['Fitness'] == best_fitness['Fitness'].max()]
        best_restart = tmp['Restarts'].min()
        best_fitness = best_fitness[best_fitness['current_restart']== best_restart]
    print('\n\n', best_state)
    # Output results
    print(f"\t\tBest Fitness: {best_fitness}")
    best_fitness.to_csv(f'./results/experiment{args.experiment}_{args.algo}.csv')

if __name__=='__main__':
    main()