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
    algo_success = {}  # Speichert Erfolgsmittelwerte für jeden Algorithmus
    algo_hops = {}     # Speichert Hops-Mittelwerte für jeden Algorithmus

    files = sorted([f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-FR") and f.endswith(".txt")],
                   key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Einzelne FR-Dateien verarbeiten und detaillierte Anzeige ausgeben
    for filename in files:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            print(f"\nVerarbeite Datei: {filename}")
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                avg_hops = algo_data['hops'].mean()
                avg_success = algo_data['success'].mean()

                # Ausgabe für jede FR-Datei und Algorithmus
                print(f"Algorithmus: {algo}")
                print(f"- Durchschnitt Hops: {avg_hops:.2f}")
                print(f"- Durchschnitt Resilienz (Success): {avg_success:.2f}")
                print("--------------------------------------------------")

                # Initialisiere Arrays für den Algorithmus, falls nicht vorhanden
                if algo not in algo_success:
                    algo_success[algo] = []
                    algo_hops[algo] = []

                # Mittelwerte für die aktuelle FR-Datei zum jeweiligen Array hinzufügen
                algo_success[algo].append(avg_success)
                algo_hops[algo].append(avg_hops)

    # Am Ende die Ergebnisarrays ausgeben und plotten
    print("\nBerechnungen von Durchschnittswerten pro Algorithmus über alle FR-Dateien:")
    for algo in algo_success:
        print(f"{algo}_success = {algo_success[algo]}")
        print(f"{algo}_hops = {algo_hops[algo]}")

    # Plots erstellen
    plot_results(algo_success, algo_hops)

def plot_results(algo_success, algo_hops):
    # Plot für Success-Werte
    plt.figure(figsize=(12, 6))
    for algo, values in algo_success.items():
        plt.plot(values, label=f"{algo} Success")
    plt.title("Durchschnittlicher Success pro FR-Datei für jeden Algorithmus")
    plt.xlabel("FR-Datei Index")
    plt.ylabel("Durchschnittlicher Success")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot für Hops-Werte
    plt.figure(figsize=(12, 6))
    for algo, values in algo_hops.items():
        plt.plot(values, label=f"{algo} Hops")
    plt.title("Durchschnittliche Hops pro FR-Datei für jeden Algorithmus")
    plt.xlabel("FR-Datei Index")
    plt.ylabel("Durchschnittliche Hops")
    plt.legend()
    plt.grid(True)
    plt.show()

# Beispiel: Verzeichnis 'results' durchsuchen
directory = 'results'
calculate_averages(directory)
