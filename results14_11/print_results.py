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
    failed_frs = set() # Speichert die FR-Nummern, bei denen mindestens ein Algorithmus success == 0 hatte

    files = sorted([f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-RANDOM-FR") and f.endswith(".txt")],
                   key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Einzelne FR-Dateien verarbeiten und detaillierte Anzeige ausgeben
    for filename in files:
        filepath = os.path.join(directory, filename)
        fr_number = int(''.join(filter(str.isdigit, filename)))  # Extrahiere die FR-Nummer
        df = process_results_file(filepath)
        if df is not None:
            print(f"\nVerarbeite Datei: {filename}")
            fr_has_zero_success = False  # Überprüft, ob mindestens ein Algorithmus in dieser FR-Datei success == 0 hat
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

                # Wenn avg_success == 0, markiere diese FR-Datei
                if avg_success == 0:
                    fr_has_zero_success = True

            # Falls mindestens ein Algorithmus success == 0 hatte, füge die FR-Nummer hinzu
            if fr_has_zero_success:
                failed_frs.add(fr_number)

    # Am Ende die Ergebnisarrays ausgeben und plotten
    print("\nBerechnungen von Durchschnittswerten pro Algorithmus über alle FR-Dateien:")
    for algo in algo_success:
        print(f"{algo}_success = {algo_success[algo]}")
        print(f"{algo}_hops = {algo_hops[algo]}")

    # Ausgabe der FR-Nummern mit success == 0
    print("\nFR-Nummern, bei denen der Success-Wert 0 war:")
    print(sorted(failed_frs))

    # Plots erstellen
    plot_results(algo_success, algo_hops)

def plot_results(algo_success, algo_hops):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot für Success-Werte
    for algo, values in algo_success.items():
        ax1.plot(values, label=f"{algo} Success")
    ax1.set_title("Durchschnittlicher Success pro FR-Datei für jeden Algorithmus")
    ax1.set_xlabel("FR-Datei Index")
    ax1.set_ylabel("Durchschnittlicher Success")
    ax1.legend()
    ax1.grid(True)

    # Plot für Hops-Werte
    for algo, values in algo_hops.items():
        ax2.plot(values, label=f"{algo} Hops")
    ax2.set_title("Durchschnittliche Hops pro FR-Datei für jeden Algorithmus")
    ax2.set_xlabel("FR-Datei Index")
    ax2.set_ylabel("Durchschnittliche Hops")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Beispiel: Verzeichnis 'results' durchsuchen
directory = 'results14_11'
calculate_averages(directory)
