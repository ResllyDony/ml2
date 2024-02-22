import mlrose_hiive as mlrose
import numpy as np
import networkx as nx
import random
import argparse

def experiment1():
   # Create a random graph with 100 nodes
    G = nx.random_geometric_graph(70, 0.5)
    # Extract node coordinates
    coords_list = [tuple(map(float, coord)) for coord in nx.get_node_attributes(G, 'pos').values()]
    # Define distance function
    fitness = mlrose.TravellingSales(coords=coords_list)
    # Define optimization problem object
    problem = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness, maximize=False)
    return problem

def experiment2():
    # Define fitness function
    fitness = mlrose.FourPeaks(t_pct=0.15)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment3():
    # Define fitness function
    fitness = mlrose.FourPeaks(t_pct=0.15)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    return problem


def experiment4():
    G = nx.gnp_random_graph(n=20, p=0.5)
    edges = list(G.edges())
    
    # Define the Max-K Color fitness function
    fitness = mlrose.MaxKColor(edges)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=len(edges), fitness_fn=fitness, maximize=False, max_val=15)
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
        "Random_Hill": mlrose.random_hill_climb,
        "Simulated_Annealing": mlrose.simulated_annealing,
        "Genetic_Algorithm": mlrose.genetic_alg,
        "MIMIC": mlrose.mimic
    }

    # Loop through problems and algorithms
    # problems = [experiment1, experiment2, experiment3, experiment4, experiment5]
    problems = [experiment1, experiment2, experiment3, experiment4, ]
    problem = problems[int(args.experiment)-1]()
    print(f"Problem {args.experiment}: {problem.__class__.__name__}")
    

    print(f"\tAlgorithm: {args.algo}")
    alg_func=algorithms[args.algo]
    # Run algorithm
    result = alg_func(problem, random_state=69, max_attempts=100, max_iters=1000)

    if isinstance(result, tuple) and len(result) == 3:
        best_state, best_fitness, _ = result
        print(f"\t\tBest Fitness: {best_fitness}")
    if isinstance(result, tuple) and len(result) == 2:
        best_state, best_fitness = result
        print(f"\t\tBest Fitness: {best_fitness}")
    
    # Output results
    # print(f"\t\tBest Fitness: {best_fitness}")

if __name__=='__main__':
    main()