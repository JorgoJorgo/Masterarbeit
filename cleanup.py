import os
import glob

def clean_directories():
    # Basisverzeichnis für die Masterarbeit
    current_dir = os.getcwd()
    
    # Prüfen, ob das Skript im Masterarbeit-Ordner liegt
    if "Masterarbeit" not in current_dir:
        print("Das Skript muss im 'Masterarbeit'-Ordner oder darunter ausgeführt werden.")
        return
    
    # Verzeichnis "results" und .txt-Dateien entfernen
    results_dir = os.path.join(current_dir, "results")
    if os.path.exists(results_dir):
        txt_files = glob.glob(os.path.join(results_dir, "*.txt"))
        for file in txt_files:
            os.remove(file)
        print(f"Entfernte Dateien im Verzeichnis {results_dir}: {txt_files}")
    else:
        print(f"Verzeichnis {results_dir} existiert nicht.")
    
    # Verzeichnis "failedgraphs" und alle Dateien entfernen
    failedgraphs_dir = os.path.join(current_dir, "failedgraphs")
    if os.path.exists(failedgraphs_dir):
        all_files = glob.glob(os.path.join(failedgraphs_dir, "*"))
        for file in all_files:
            os.remove(file)
        print(f"Entfernte Dateien im Verzeichnis {failedgraphs_dir}: {all_files}")
    else:
        print(f"Verzeichnis {failedgraphs_dir} existiert nicht.")

    # Verzeichnis "graphen" und alle Dateien entfernen
    failedgraphs_dir = os.path.join(current_dir, "graphen")
    if os.path.exists(failedgraphs_dir):
        all_files = glob.glob(os.path.join(failedgraphs_dir, "*.png"))
        for file in all_files:
            os.remove(file)
        print(f"Entfernte Dateien im Verzeichnis {failedgraphs_dir}: {all_files}")
    else:
        print(f"Verzeichnis {failedgraphs_dir} existiert nicht.")

if __name__ == "__main__":
    clean_directories()
