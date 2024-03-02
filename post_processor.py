import pandas as pd
import matplotlib.pyplot as plt
import os

directory = './results'

e1={'name': "Flip Flop"}
e2={'name': "4 Peaks"}
e3={'name': "Knapsack"}
experiments = [e1, e2, e3]

for f in os.listdir(directory):
    data = pd.read_csv(f'{directory}/{f}')
    algo_name = f.split('_')[1]
    algo_name = os.path.splitext(algo_name)[0]
    if f.startswith("experiment1"):
        e1[algo_name] = data
    elif f.startswith('experiment2'):
        e2[algo_name] = data
    elif f.startswith('experiment3'):
        e3[algo_name] = data

for i, experiment in enumerate(experiments):
    algo_list=[]
    for algo, data in experiment.items():
        if algo=='name' or algo=='sa':
            continue
        plt.plot(data['Iteration'], data['Fitness'])
        algo_list.append(algo)
    plt.xlabel('Iteration')  # Add x-axis label
    plt.ylabel('Fitness')    # Add y-axis label
    plt.title(f'Experiment {i+1}: {experiment["name"]}')  # Add figure title
    plt.legend(algo_list)
    plt.savefig(f'figures/experiment{i+1}_iterations.png')
    plt.clf()

    algo_list=[]
    for algo, data in experiment.items():
        if algo=='name' or (algo=='rhc' and i==0) or algo=='sa':
            continue
        plt.plot(data['FEvals'], data['Fitness'])
        algo_list.append(algo)
    plt.xlabel('FEvals')  # Add x-axis label
    plt.ylabel('Fitness')    # Add y-axis label
    plt.title(f'Experiment {i+1}: {experiment["name"]}')  # Add figure title
    plt.legend(algo_list)
    plt.savefig(f'figures/experiment{i+1}_fevals.png')
    plt.clf()

    algo_list=[]
    time_list=[]
    for j, (algo, data) in enumerate(experiment.items()):
        if algo == 'name':
            continue
        best_fit = data['Fitness'].max()
        best_fit_rows = data[data['Fitness']==best_fit]
        best_time = best_fit_rows['Time'].min()
        print(type(best_time))
        plt.bar(algo, best_time)  # Plot a bar for each algorithm
        algo_list.append(algo)
        
        time_list.append(best_time)
        # Annotate each bar with its time value
    for x, y in zip(algo_list,time_list):
        print(y)
        plt.text(x, y, f'{round(y, 5)}', ha='center', va='bottom') 

    plt.xlabel('Algorithm')   # Add x-axis label
    plt.ylabel('Time')        # Add y-axis label
    plt.title(f'Experiment {i+1}: {experiment["name"]}')  # Add figure title
    plt.legend(algo_list)     # Add legend
    plt.savefig(f'figures/experiment{i+1}_time.png')
    plt.clf()