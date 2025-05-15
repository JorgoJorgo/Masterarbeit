import os
import pandas as pd
import matplotlib.pyplot as plt

# Stilkonfiguration für alle Algorithmen
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

# === Funktion zum Einlesen der Resilienzdaten ===
def load_resilience_data(base_path, folders):
    records = []
    for folder in folders:
        full_path = os.path.join(base_path, folder)
        for file in os.listdir(full_path):
            if file.endswith(".txt") and "-FR" in file:
                fr = int(file.split("-FR")[-1].replace(".txt", ""))
                with open(os.path.join(full_path, file), 'r') as f:
                    for line in f:
                        if line.startswith("#") or line.strip() == "":
                            continue
                        parts = line.strip().split(",")
                        if len(parts) >= 9:
                            algorithm = parts[3].strip()
                            resilience = float(parts[8])
                            records.append((folder, algorithm, fr, resilience))
    df = pd.DataFrame(records, columns=["scenario", "algorithm", "failure_rate", "resilience"])
    return df

# === Plotfunktion für Resilienz ===
def plot_resilience(df, label, failure_max=20):
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
                    alg_df["resilience"],
                    label=alg,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2
                )

        ax.set_title(f"n = {size}", fontsize=12)
        ax.set_xlabel("Failure Rate")
        ax.set_xlim(0, failure_max)
        ax.set_xticks(range(0, failure_max + 1, 5))
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        if size == 49:
            ax.set_ylabel("Resilienz")
    if label == "random":
        title = f"Resilienzwerte für Real-World-Graphen mit randomisierten Fehlern"
    else:
        title = f"Resilienzwerte für Real-World-Graphen mit geclusterten Fehlern"
    fig.suptitle(title, fontsize=14)
    fig.legend(style_config.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
    plt.show()

# === Hauptprogramm ===
if __name__ == "__main__":
    base_dir = "./"  # <- ggf. anpassen
    folders = ["49-random", "54-random", "57-random", "49-cluster", "54-cluster", "57-cluster"]
    df = load_resilience_data(base_dir, folders)

    # Erst: randomisierte Fehler
    plot_resilience(df, label="random")

    # Danach: geclusterte Fehler
    plot_resilience(df, label="cluster")
