import os
import glob

def clean_directories():
    # Verzeichnis "results" aufrufen und .txt-Dateien entfernen
    results_dir = os.path.expanduser("~/Desktop/Masterarbeit/results")
    if os.path.exists(results_dir):
        os.chdir(results_dir)
        txt_files = glob.glob("*.txt")
        for file in txt_files:
            os.remove(file)
        print(f"Entfernte Dateien im Verzeichnis {results_dir}: {txt_files}")
    else:
        print(f"Verzeichnis {results_dir} existiert nicht.")

    # Zurück ins Hauptverzeichnis wechseln
    masterarbeit_dir = os.path.expanduser("~/Desktop/Masterarbeit")
    os.chdir(masterarbeit_dir)

    # Verzeichnis "failedgraphs" aufrufen und alle Dateien entfernen
    failedgraphs_dir = os.path.join(masterarbeit_dir, "failedgraphs")
    if os.path.exists(failedgraphs_dir):
        os.chdir(failedgraphs_dir)
        all_files = glob.glob("*")
        for file in all_files:
            os.remove(file)
        print(f"Entfernte Dateien im Verzeichnis {failedgraphs_dir}: {all_files}")
    else:
        print(f"Verzeichnis {failedgraphs_dir} existiert nicht.")

    # Zurück ins Hauptverzeichnis wechseln
    os.chdir(masterarbeit_dir)

if __name__ == "__main__":
    clean_directories()
