import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

        # Ersetze 'inf' und float('inf') durch 0
        df.replace(['inf', float('inf')], 0, inplace=True)
        
        # Sicherstellen, dass alle relevanten Spalten numerisch sind
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
        [f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-RANDOM-FR") and f.endswith(".txt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    found_algorithms = set()

    for filename in files:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            for algo in df['algorithm'].unique():
                found_algorithms.add(algo)
                if selected_algorithms is None or algo in selected_algorithms:
                    algo_data = df[df['algorithm'] == algo]
                    avg_hops = algo_data['hops'].mean()
                    avg_success = algo_data['success'].mean()

                    if algo not in algo_success:
                        algo_success[algo] = []
                        algo_hops[algo] = []

                    algo_success[algo].append(avg_success)
                    algo_hops[algo].append(avg_hops)
    
    print("Gefundene Algorithmen in den Dateien:", found_algorithms)

    if selected_algorithms:
        missing_algorithms = set(selected_algorithms) - found_algorithms
        if missing_algorithms:
            print(f"Warnung: Die folgenden ausgewählten Algorithmen wurden in den Daten nicht gefunden: {missing_algorithms}")

    plot_success(title_prefix="RANDOM", algo_success=algo_success)
    plot_hops(title_prefix="RANDOM", algo_hops=algo_hops)

def generate_colors(num_colors):
    """
    Erzeugt eine Liste von Farben aus der 'tab10'-Palette oder zufällig generierte Farben,
    falls mehr als 10 Farben benötigt werden.
    """
    if num_colors <= 10:
        return plt.cm.tab10.colors[:num_colors]  # Verwende die vordefinierten Farben
    else:
        np.random.seed(42)  # Feste Zufallswerte für Konsistenz
        return [np.random.rand(3,) for _ in range(num_colors)]  # Zufällige Farben

def plot_success(title_prefix, algo_success):
    plt.figure(figsize=(14, 6))
    if algo_success:
        colors = generate_colors(len(algo_success))
        linestyles = ['-', '--', '-.', ':']  # Verschiedene Linienstile für Unterscheidung
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']  # Unterschiedliche Marker

        for i, (algo, values) in enumerate(algo_success.items()):
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(values, label=f"{algo} Success", color=colors[i], linestyle=linestyle, marker=marker, alpha=0.8)

        plt.title(f"{title_prefix} - Durchschnittlicher Success pro FR-Datei")
        plt.xlabel("FR-Datei Index")
        plt.ylabel("Durchschnittlicher Success")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print("Keine Daten für Success-Werte vorhanden.")

def plot_hops(title_prefix, algo_hops):
    plt.figure(figsize=(14, 6))
    if algo_hops:
        colors = generate_colors(len(algo_hops))
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'D', '^', 'v', 'x', '*', 'P', 'H', '8']

        for i, (algo, values) in enumerate(algo_hops.items()):
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(values, label=f"{algo} Hops", color=colors[i], linestyle=linestyle, marker=marker, alpha=0.8)

        plt.title(f"{title_prefix} - Durchschnittliche Hops pro FR-Datei")
        plt.xlabel("FR-Datei Index")
        plt.ylabel("Durchschnittliche Hops")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
    else:
        print("Keine Daten für Hops-Werte vorhanden.")

# Beispielaufruf:
directory = 'results'
selected_algorithms = [
    #" MaxDAG", 
    #" MultipleTrees Random Checkpoint", 
    #" MultipleTrees Random Checkpoint Parallel", 
    #" MultipleTrees Closeness Checkpoint", 
    #" MultipleTrees Betweenness Checkpoint", 
    #" MultipleTrees Degree Checkpoint", 
    #' MultipleTrees Inverse Degree Checkpoint',
    #' MultipleTrees Inverse Degree Greedy Checkpoint',
    
    #" One Tree Middle Checkpoint PE", 
    #" One Tree Closeness Checkpoint PE",
    #" One Tree Degree Checkpoint PE", 
    #" One Tree Betweenness Checkpoint PE", 
    " One Tree Shortest EDP Checkpoint PE", 
    
    " Triple Checkpoint OneTree", 
    #" Triple Checkpoint MultipleTrees", 
    
    " SquareOne Cuts", 
    " MultipleTrees Cuts",
    #" MultipleTrees Faces",
]
calculate_averages(directory, selected_algorithms)

# Falls alle Algorithmen verwendet werden sollen:
calculate_averages(directory)
