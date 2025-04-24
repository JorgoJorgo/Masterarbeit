import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        [f for f in os.listdir(directory) if f.startswith("benchmark-planar-gabriel-CLUSTER-FR") and f.endswith(".txt")],
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
            print(f"Warnung: Diese ausgewählten Algorithmen wurden in den Daten nicht gefunden: {missing_algorithms}")

    print("\nLängen der Success-Listen pro Algorithmus:")
    for algo, values in algo_success.items():
        print(f"- {algo}: {len(values)} Werte")

    fr_indices = [int(re.search(r'FR(\d+)', f).group(1)) for f in files]

    plot_success(title_prefix="RANDOM", algo_success=algo_success, fr_indices=fr_indices)
    plot_hops(title_prefix="RANDOM", algo_hops=algo_hops, fr_indices=fr_indices)

def generate_colors(num_colors):
    if num_colors <= 10:
        return plt.cm.tab10.colors[:num_colors]
    else:
        np.random.seed(42)
        return [np.random.rand(3,) for _ in range(num_colors)]

def plot_success(title_prefix, algo_success, fr_indices):
    plt.figure(figsize=(14, 6))
    if algo_success:
        colors = generate_colors(len(algo_success))
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']
        num_points = len(fr_indices)

        for i, (algo, values) in enumerate(algo_success.items()):
            if len(values) < num_points:
                print(f"Werte für {algo} unvollständig: {len(values)} → auffüllen auf {num_points}")
                values += [np.nan] * (num_points - len(values))
            elif len(values) > num_points:
                print(f"Werte für {algo} zu lang: {len(values)} → kürzen auf {num_points}")
                values = values[:num_points]

            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(fr_indices, values, label=f"{algo.strip()} Success", color=colors[i], linestyle=linestyle, marker=marker, alpha=0.8)

        plt.title(f"{title_prefix} - Durchschnittlicher Success pro FR-Datei")
        plt.xlabel("FR-Datei Index")
        plt.ylabel("Durchschnittlicher Success")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print("Keine Daten für Success-Werte vorhanden.")

def plot_hops(title_prefix, algo_hops, fr_indices):
    plt.figure(figsize=(14, 6))
    if algo_hops:
        colors = generate_colors(len(algo_hops))
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']
        num_points = len(fr_indices)

        for i, (algo, values) in enumerate(algo_hops.items()):
            if len(values) < num_points:
                print(f"Werte für {algo} unvollständig: {len(values)} → auffüllen auf {num_points}")
                values += [np.nan] * (num_points - len(values))
            elif len(values) > num_points:
                print(f"Werte für {algo} zu lang: {len(values)} → kürzen auf {num_points}")
                values = values[:num_points]

            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(fr_indices, values, label=f"{algo.strip()} Hops", color=colors[i], linestyle=linestyle, marker=marker, alpha=0.8)

        plt.title(f"{title_prefix} - Durchschnittliche Hops pro FR-Datei")
        plt.xlabel("FR-Datei Index")
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
   ' SquareOne Cuts',
   ' MultipleTrees Degree Checkpoint Extended',
   ' MultipleTrees Inverse Degree Checkpoint Extended',
   ' MultipleTrees Cuts Extended',
   ' MultipleTrees Faces Extended',
]
calculate_averages(directory, selected_algorithms)

# Falls alle Algorithmen verwendet werden sollen:
calculate_averages(directory)
