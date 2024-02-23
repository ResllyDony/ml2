import mlrose_hiive as mlrose
import numpy as np

# Define functions to create optimization problems
def experiment1():
    # Define fitness function
    fitness = mlrose.OneMax()
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def experiment2():
    # Create a list of city coordinates
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
    # Define distance function
    fitness = mlrose.TravellingSales(coords=coords_list)
    # Define optimization problem object
    problem = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness, maximize=False)
    return problem

def continuous_peaks_fitness(state):
    fitness = 0
    for i in range(len(state) - 1):
        if state[i] == 1:
            fitness += 1
            if state[i] == state[i + 1]:
                fitness += 1
    return fitness

def experiment3():
    # Define fitness function
    fitness = mlrose.CustomFitness(continuous_peaks_fitness)
    # Define optimization problem object
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    return problem

def main():
    # Define optimization algorithms
    algorithms = {
        "Random Hill Climbing": mlrose.random_hill_climb,
        "Simulated Annealing": mlrose.simulated_annealing,
        "Genetic Algorithm": mlrose.genetic_alg,
        "MIMIC": mlrose.mimic
    }

    # Loop through problems and algorithms
    problems = [experiment1, experiment2, experiment3]

    for problem_func in problems:
        problem = problem_func()
        print(f"Problem: {problem.__class__.__name__}")
        
        for alg_name, alg_func in algorithms.items():
            print(f"\tAlgorithm: {alg_name}")
            
            # Run algorithm
            # best_state, best_fitness = alg_func(problem)
            result = alg_func(problem)

            if isinstance(result, tuple) and len(result) == 2:
                best_state, best_fitness = result
                print(f"\t\tBest Fitness: {best_fitness}")
            else:
                print("Unexpected return value:", result)
            
            # Output results
            # print(f"\t\tBest Fitness: {best_fitness}")

if __name__=='__main__':
    main()
