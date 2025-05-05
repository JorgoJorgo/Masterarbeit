import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import re

def process_results_file(filename):
    try:
        df = pd.read_csv(
            filename,
            comment='#',
            header=None,
            names=[
                'graph', 'size', 'connectivity', 'algorithm', 'index',
                'stretch', 'load', 'hops', 'success',
                'routing computation time', 'pre-computation time in seconds'
            ]
        )

        df.replace(['inf', float('inf')], 0, inplace=True)

        for col in ['stretch', 'load', 'hops', 'success', 'routing computation time', 'pre-computation time in seconds']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {filename}: {e}")
        return None

def calculate_averages(directory, selected_algorithms=None):
    algo_success = {}
    algo_hops = {}

    files = sorted(
        [f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-CLUSTER-FR") and f.endswith(".txt")],
        key=lambda x: int(re.search(r'FR(\d+)', x).group(1))
    )

    print(f"Gefundene Dateien ({len(files)}):", files)

    found_algorithms = set()

    if selected_algorithms:
        for algo in selected_algorithms:
            algo_success[algo] = []
            algo_hops[algo] = []

    for filename in files:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            algos_in_file = set(df['algorithm'].unique())
            found_algorithms.update(algos_in_file)

            if selected_algorithms is None:
                for algo in algos_in_file:
                    if algo not in algo_success:
                        algo_success[algo] = []
                        algo_hops[algo] = []

                current_algos = algos_in_file
            else:
                current_algos = selected_algorithms

            for algo in current_algos:
                if algo in algos_in_file:
                    algo_data = df[df['algorithm'] == algo]
                    avg_hops = algo_data['hops'].mean()
                    avg_success = algo_data['success'].mean()
                else:
                    avg_hops = np.nan
                    avg_success = np.nan

                algo_success[algo].append(avg_success)
                algo_hops[algo].append(avg_hops)

    print("\nGefundene Algorithmen in den Dateien:", found_algorithms)

    if selected_algorithms:
        missing_algorithms = set(selected_algorithms) - found_algorithms
        if missing_algorithms:
            print(f" Warnung: Diese ausgewählten Algorithmen wurden in den Daten nicht gefunden: {missing_algorithms}")

    print("\nLängen der Success-Listen pro Algorithmus:")
    for algo, values in algo_success.items():
        print(f"- {algo}: {len(values)} Werte")

    fr_indices = [int(re.search(r'FR(\d+)', f).group(1)) for f in files]

    plot_success(title_prefix="RANDOM", algo_success=algo_success, fr_indices=fr_indices)
    plot_hops(title_prefix="RANDOM", algo_hops=algo_hops, fr_indices=fr_indices)

# Feste Farbzuteilung
def get_algorithm_color_mapping():
    return {
        ' One Tree Middle Checkpoint PE': 'red',
        ' MaxDAG': 'blue',
        ' SquareOne': 'green'
    }

def generate_colors(num_colors):
    base_palette = cm.get_cmap('Set1', num_colors)
    return [base_palette(i) for i in range(num_colors)]

def plot_success(title_prefix, algo_success, fr_indices):
    plt.figure(figsize=(14, 6))
    if algo_success:
        colors = generate_colors(len(algo_success))
        color_mapping = get_algorithm_color_mapping()
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']

        for i, (algo, values) in enumerate(algo_success.items()):
            color = color_mapping.get(algo, colors[i])
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(fr_indices, values, label=f"{algo.strip()}", color=color, linestyle=linestyle, marker=marker, alpha=0.9)

        plt.title("Durchschnittliche Resilienz pro Failure Rate (Delaunay, Geclusterte Fehler)")
        plt.xlabel("Failure Rate")
        plt.ylabel("Resilienz")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print("Keine Daten für Success-Werte vorhanden.")

def plot_hops(title_prefix, algo_hops, fr_indices):
    plt.figure(figsize=(14, 6))
    if algo_hops:
        colors = generate_colors(len(algo_hops))
        color_mapping = get_algorithm_color_mapping()
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']

        for i, (algo, values) in enumerate(algo_hops.items()):
            color = color_mapping.get(algo, colors[i])
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(fr_indices, values, label=f"{algo.strip()}", color=color, linestyle=linestyle, marker=marker, alpha=0.9)

        plt.title("Durchschnittliche Hops pro Failure Rate (Delaunay, Geclusterte Fehler)")
        plt.xlabel("Failure Rate")
        plt.ylabel("Durchschnittliche Hops")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print("Keine Daten für Hops-Werte vorhanden.")

# Beispielaufruf:
directory = 'results_cluster_delaunay'

selected_algorithms = [
   ' MaxDAG',
   ' SquareOne',
   ' MultipleTrees Betweenness Checkpoint',
   ' MultipleTrees Inverse Middle Greedy Checkpoint',
   ' MultipleTrees Random Checkpoint',
   ' MultipleTrees Cuts Extended',
   ' MultipleTrees Faces Extended',
   ' One Tree Middle Checkpoint PE',
]

calculate_averages(directory, selected_algorithms)

# Falls alle Algorithmen verwendet werden sollen:
calculate_averages(directory)
