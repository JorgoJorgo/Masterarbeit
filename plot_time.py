import os
import pandas as pd
import matplotlib.pyplot as plt

# Relevante Algorithmen + Farben
relevant_algorithms = [
    "MaxDAG",
    "SquareOne",
    "One Tree Middle Checkpoint PE",
    "MultipleTrees Betweenness Checkpoint",
    "MultipleTrees Inverse Middle Greedy Checkpoint",
    "MultipleTrees Random Checkpoint",
    "MultipleTrees Cuts Extended",
    "MultipleTrees Faces Extended",
    "Triple Checkpoint MultipleTrees"
]

algorithm_colors = {
    "MaxDAG": "blue",
    "SquareOne": "green",
    "One Tree Middle Checkpoint PE": "red",
    "MultipleTrees Betweenness Checkpoint": "limegreen",
    "MultipleTrees Inverse Middle Greedy Checkpoint": "orchid",
    "MultipleTrees Random Checkpoint": "gold",
    "MultipleTrees Cuts Extended": "brown",
    "MultipleTrees Faces Extended": "hotpink",
    "Triple Checkpoint MultipleTrees": "darkorange"
}

# === Funktion: Dateien einlesen (routing_time + precomp_time summieren) ===
def load_runtime_data(directory):
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and "-FR" in filename:
            with open(os.path.join(directory, filename), "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split(",")
                    if len(parts) >= 11:
                        algorithm = parts[3].strip()
                        try:
                            routing_time = float(parts[9])
                            precomp_time = float(parts[10])
                            total_time = routing_time + precomp_time
                            records.append((algorithm, total_time))
                        except ValueError:
                            continue
    return pd.DataFrame(records, columns=["algorithm", "total_time"])

# === Funktion: Farbkodierter Boxplot mit MaxDAG + SquareOne Labels ===
def plot_colored_boxplot_partial_labels(df, title):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    algs = sorted(df['algorithm'].unique(), key=lambda x: relevant_algorithms.index(x))
    data = [df[df['algorithm'] == alg]['total_time'] for alg in algs]
    box = ax.boxplot(data, patch_artist=True)

    for patch, alg in zip(box['boxes'], algs):
        patch.set_facecolor(algorithm_colors[alg])

    # Nur MaxDAG und SquareOne beschriften
    xtick_labels = ['' for _ in algs]
    for i, alg in enumerate(algs):
        if alg in ["MaxDAG", "SquareOne"]:
            xtick_labels[i] = alg
    ax.set_xticks(range(1, len(algs) + 1))
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    ax.set_ylabel("Rechenzeit (s)")
    ax.set_title(title)

    handles = [plt.Line2D([0], [0], color=algorithm_colors[alg], lw=10) for alg in algs]
    ax.legend(handles, algs, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    plt.tight_layout()
    plt.show()

# === HAUPTTEIL ===
if __name__ == "__main__":
    # Pfade anpassen je nach Projektstruktur
    base_random_dirs = [
        "./results_random_delaunay",
        "./results_random_gabriel",
        "./results_cluster_delaunay",
        "./results_cluster_gabriel"
    ]

    base_real_dirs = [
        "./49-random",
        "./54-random",
        "./57-random",
        "./49-cluster",
        "./54-cluster",
        "./57-cluster"
    ]

    # Daten aggregieren
    df_random = pd.concat([load_runtime_data(p) for p in base_random_dirs], ignore_index=True)
    df_real = pd.concat([load_runtime_data(p) for p in base_real_dirs], ignore_index=True)

    # Filter relevante Algorithmen
    df_random_filtered = df_random[df_random["algorithm"].isin(relevant_algorithms)]
    df_real_filtered = df_real[df_real["algorithm"].isin(relevant_algorithms)]

    # Plots erzeugen
    plot_colored_boxplot_partial_labels(df_random_filtered, "Rechenzeit auf randomisierten Graphen")
    plot_colored_boxplot_partial_labels(df_real_filtered, "Rechenzeit auf Real-World-Graphen")
