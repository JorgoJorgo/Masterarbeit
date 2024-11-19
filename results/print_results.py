import os
import pandas as pd
import matplotlib.pyplot as plt

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

def calculate_averages(directory):
    algo_success_cluster = {}
    algo_hops_cluster = {}
    algo_success_random = {}
    algo_hops_random = {}

    # Unterstützt sowohl RANDOM- als auch CLUSTER-Dateien
    files_cluster = sorted(
        [f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-CLUSTER-FR") and f.endswith(".txt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    files_random = sorted(
        [f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-RANDOM-FR") and f.endswith(".txt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    # Verarbeite CLUSTER-Dateien
    for filename in files_cluster:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                avg_hops = algo_data['hops'].mean()
                avg_success = algo_data['success'].mean()

                if algo not in algo_success_cluster:
                    algo_success_cluster[algo] = []
                    algo_hops_cluster[algo] = []

                algo_success_cluster[algo].append(avg_success)
                algo_hops_cluster[algo].append(avg_hops)

    # Verarbeite RANDOM-Dateien
    for filename in files_random:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                avg_hops = algo_data['hops'].mean()
                avg_success = algo_data['success'].mean()

                if algo not in algo_success_random:
                    algo_success_random[algo] = []
                    algo_hops_random[algo] = []

                algo_success_random[algo].append(avg_success)
                algo_hops_random[algo].append(avg_hops)

    # Plots erstellen
    plot_results("CLUSTER", algo_success_cluster, algo_hops_cluster)
    plot_results("RANDOM", algo_success_random, algo_hops_random)

def plot_results(title_prefix, algo_success, algo_hops):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot für Success-Werte
    for algo, values in algo_success.items():
        ax1.plot(values, label=f"{algo} Success")
    ax1.set_title(f"{title_prefix} - Durchschnittlicher Success pro FR-Datei")
    ax1.set_xlabel("FR-Datei Index")
    ax1.set_ylabel("Durchschnittlicher Success")
    ax1.legend()
    ax1.grid(True)

    # Plot für Hops-Werte
    for algo, values in algo_hops.items():
        ax2.plot(values, label=f"{algo} Hops")
    ax2.set_title(f"{title_prefix} - Durchschnittliche Hops pro FR-Datei")
    ax2.set_xlabel("FR-Datei Index")
    ax2.set_ylabel("Durchschnittliche Hops")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Beispiel: Verzeichnis 'results' durchsuchen
directory = 'results'
calculate_averages(directory)
