import os
import re
import pandas as pd
import numpy as np

def parse_data(content):
    columns = ['graph', 'size', 'connectivity', 'algorithm', 'index', 'stretch', 'load', 'hops', 'success', 'routing_computation_time', 'pre_computation_time']
    data = []
    for line in content:
        if line.startswith('#'):
            continue
        values = line.strip().split(', ')
        data.append(values)
    df = pd.DataFrame(data, columns=columns)
    # Replace 'inf' with NaN and convert columns to numeric
    df.replace('inf', np.nan, inplace=True)
    df[['stretch', 'load', 'hops', 'success', 'routing_computation_time', 'pre_computation_time']] = df[['stretch', 'load', 'hops', 'success', 'routing_computation_time', 'pre_computation_time']].apply(pd.to_numeric)
    return df

def calculate_averages(df):
    results = {}
    algorithms = df['algorithm'].unique()
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        avg_resilience = algo_data['success'].mean()
        avg_hops = algo_data['hops'].mean()
        results[algo] = {'avg_resilience': avg_resilience, 'avg_hops': avg_hops}
    return results

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def process_folder(folder_path):
    avg_resilience_array = {}
    avg_hops_array = {}

    # List and naturally sort the files to ensure correct order
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")], key=natural_sort_key)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            content = file.readlines()
            df = parse_data(content)
            averages = calculate_averages(df)
            
            for algo, metrics in averages.items():
                if algo not in avg_resilience_array:
                    avg_resilience_array[algo] = []
                    avg_hops_array[algo] = []
                
                avg_resilience_array[algo].append(metrics['avg_resilience'])
                avg_hops_array[algo].append(metrics['avg_hops'])
    
    return avg_resilience_array, avg_hops_array

# Example usage
folder_path = './'
avg_resilience_array, avg_hops_array = process_folder(folder_path)

print("Average Resilience:", avg_resilience_array)
print("Average Hops:", avg_hops_array)
