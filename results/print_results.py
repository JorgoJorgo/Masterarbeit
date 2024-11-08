import os
import pandas as pd

def process_results_file(filename):
    print(f"Verarbeite Datei: {filename}")
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
        return df
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {filename}: {e}")
        return None

def calculate_averages(directory):
    all_data = []
    fr_results = {}

    # Dateien in numerischer Reihenfolge sortieren
    files = sorted([f for f in os.listdir(directory) if f.startswith("benchmark-planar-delaunay-FR") and f.endswith(".txt")],
                   key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Einzeln die FR Dateien verarbeiten
    for filename in files:
        filepath = os.path.join(directory, filename)
        df = process_results_file(filepath)
        if df is not None:
            all_data.append(df)
            
            # Berechnung der Durchschnittswerte pro Algorithmus für jede FR-Datei
            fr_results[filename] = {}
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                avg_hops = algo_data['hops'].mean()
                avg_success = algo_data['success'].mean()
                fr_results[filename][algo] = {'Hops': avg_hops, 'Success': avg_success}

    # Ergebnisse für jede FR-Datei anzeigen
    for filename, results in fr_results.items():
        print(f"\nErgebnisse für {filename}:")
        for algo, metrics in results.items():
            print(f"Algorithmus: {algo}")
            print(f"- Durchschnitt Hops: {metrics['Hops']:.2f}")
            print(f"- Durchschnitt Resilienz (Success): {metrics['Success']:.2f}")
            print("-----------------------------------")

    if not all_data:
        print("Keine gültigen Dateien gefunden.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Berechnung der Gesamt-Durchschnittswerte für alle Dateien
    print("\nGesamtdurchschnittswerte pro Algorithmus:\n")
    for algo in combined_df['algorithm'].unique():
        algo_data = combined_df[combined_df['algorithm'] == algo]
        print(f"Algorithmus: {algo}")
        print(f"- Durchschnitt Hops: {algo_data['hops'].mean():.2f}")
        print(f"- Durchschnitt Resilienz (Success): {algo_data['success'].mean():.2f}")
        print(f"- Durchschnitt Routing Computation Time: {algo_data['routing computation time'].mean():.6f}")
        print(f"- Durchschnitt Pre-Computation Time: {algo_data['pre-computation time in seconds'].mean():.6f}")
        print("--------------------------------------------------")

# Verzeichnis mit den Ergebnisdateien
directory = "results"
calculate_averages(directory)
