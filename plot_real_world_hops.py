import os
import pandas as pd
import matplotlib.pyplot as plt

# === Stilkonfiguration für alle Algorithmen ===
style_config = {
    "MaxDAG": {"color": "blue", "marker": "o", "linestyle": "-"},
    "SquareOne": {"color": "green", "marker": "s", "linestyle": "--"},
    "One Tree Middle Checkpoint PE": {"color": "red", "marker": "d", "linestyle": ":"},
    "MultipleTrees Betweenness Checkpoint": {"color": "limegreen", "marker": "D", "linestyle": "-."},
    "MultipleTrees Inverse Middle Greedy Checkpoint": {"color": "orchid", "marker": "^", "linestyle": ":"},
    "MultipleTrees Random Checkpoint": {"color": "gold", "marker": "*", "linestyle": "-"},
    "MultipleTrees Cuts Extended": {"color": "brown", "marker": "x", "linestyle": "--"},
    "MultipleTrees Faces Extended": {"color": "hotpink", "marker": "P", "linestyle": "-."},
    "Triple Checkpoint MultipleTrees": {"color": "darkorange", "marker": "X", "linestyle": "-"}
}

# === Funktion zum Einlesen der Hops-Daten (inf → 0.0) ===
def load_hops_data(base_path, folders):
    records = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt") and "-FR" in file:
                try:
                    fr = int(file.split("-FR")[-1].replace(".txt", ""))
                except:
                    continue
                with open(os.path.join(folder_path, file), 'r') as f:
                    for line in f:
                        if line.startswith("#") or line.strip() == "":
                            continue
                        parts = line.strip().split(",")
                        if len(parts) >= 8:
                            algorithm = parts[3].strip()
                            hop_value = parts[7].strip()
                            if "inf" in hop_value:
                                hops = 0.0
                            else:
                                try:
                                    hops = float(hop_value)
                                except ValueError:
                                    continue
                            records.append((folder, algorithm, fr, hops))
    return pd.DataFrame(records, columns=["scenario", "algorithm", "failure_rate", "hops"])

# === Plotfunktion für Hops ===
def plot_hops(df, label, failure_max=20, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    sizes = [49, 54, 57]

    for ax, size in zip(axes, sizes):
        scenario = f"{size}-{label}"
        scenario_df = df[(df["scenario"] == scenario) & (df["failure_rate"] <= failure_max)]

        for alg, style in style_config.items():
            alg_df = scenario_df[scenario_df["algorithm"] == alg].sort_values("failure_rate")
            if not alg_df.empty:
                ax.plot(
                    alg_df["failure_rate"],
                    alg_df["hops"],
                    label=alg,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2
                )

        ax.set_title(f"n = {size}", fontsize=12)
        ax.set_xlabel("FR")
        ax.set_xlim(0, failure_max)
        ax.set_xticks(range(0, failure_max + 1, 5))
        ax.grid(True)
        if size == 49:
            ax.set_ylabel("Hops")

    fig.suptitle(title, fontsize=14)
    fig.legend(style_config.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
    plt.show()

# === Hauptprogramm ===
if __name__ == "__main__":
    base_dir = "./"
    random_folders = ["49-random", "54-random", "57-random"]
    cluster_folders = ["49-cluster", "54-cluster", "57-cluster"]

    df_random = load_hops_data(base_dir, random_folders)
    df_cluster = load_hops_data(base_dir, cluster_folders)

    plot_hops(df_random, label="random", title="Durchschnittliche Hops pro Failure Rate (Real-World, Randomisierte Fehler)")
    plot_hops(df_cluster, label="cluster", title="Durchschnittliche Hops pro Failure Rate (Real-World, Geclusterte Fehler)")
