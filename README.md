# Masterarbeit

In diesem Repository befindet sich der Quellcode zur Masterarbeit „Untersuchung von Fehlerresilienz in planaren Netzwerken mit lokalen Routing-Regeln“ von Georgios Karamoussanlis. Für weiterführende Informationen empfiehlt sich ein Blick in die Readme-Datei des ursprünglichen fast failover (https://gitlab.cs.univie.ac.at/ct-papers/fast-failover)-Frameworks, auf dem diese Arbeit basiert.

## Voraussetzungen

Dieses Repository wurde unter Ubuntu 22.04 getestet. Die benötigten Python-Module können wie folgt installiert werden:
```
pip install networkx==3.2 numpy matplotlib pydot
``` 
## Übersicht

- ```trees.py```: Enthält alle Algorithmen zur Baumstruktur-Erzeugung sowie zugehörige Hilfsfunktionen.
- ```trees_with_cp.py```: Variante der Baumstrukturierung unter Nutzung von Checkpoints und Kantenflächen.
- ```routing.py```: Beinhaltet die Routing-Algorithmen.
- ```benchmark_graphs```: Verzeichnis mit den verwendeten Topologien.
- ```results```: Enthält die Ergebnisse und Ausgaben der Algorithmen.
- ```..._experiments.py```: Skripte zur Durchführung vordefinierter Experimente.
- Die Ergebnisse sind nach Fehlerraten sortiert in Unterordnern abgelegt (inkl. Benchmarks, Logs, usw.).
- ```benchmark-....txt```: Für jede Fehlerrate vorhanden. Diese Dateien können durch Anpassung der Pfade in plotter.py verwendet werden.

Die verwendeten Netzwerktopologien können von Rocketfuel (https://research.cs.washington.edu/networking/rocketfuel/) und Internet Topology Zoo (http://www.topology-zoo.org/) heruntergeladen und im Verzeichnis benchmark_graphs abgelegt werden.

## Durchführung mit zufällig generierten Graphen

Zum Starten eines Experiments mit zufällig generierten planaren Graphen kann folgender Befehl ausgeführt werden:
```
python3 planar_experiments.py planar 3 1 45 15 RANDOM
```
Bedeutung der Parameter:

- ```planar```: Gibt an, welche Experimentgruppe gestartet wird.
- ```3```: Zufalls-Seed für die Auswahl von Quelle und Ziel.
- ```1```: Anzahl der zu startenden Experimente.
- ```45```: Anzahl der Knoten im Graphen.
- ```15```: Anzahl der Quellknoten, von denen ein Paket zum Ziel geleitet wird.
- ```RANDOM```: Art des Fehlers (alternativ: ```CLUSTER```)

Der Algorithmus erzeugt Graphen mit n Knoten und versucht, diese mittels Delaunay-Triangulation oder Gabriel-Graph planar zu gestalten. Die verwendete Methode ist in der Funktion ```experiments()``` definiert. Über die Schleife ```for i in range(1,13):``` in der Main-Funktion kann die Zahl der Ausfälle gesteuert werden. In jedem Schleifendurchlauf werden zusätzlich ```f_num``` Kanten als fehlerhaft markiert.

## Durchführung mit realen Topologien

Zur Durchführung von Experimenten mit realen Netzwerktopologien aus dem Topology Zoo (inkl. Cluster-Ausfällen):
```
python3 planar_experiments.py zoo 45 5 100 5 RANDOM
```
Bedeutung der Parameter:

- ```zoo```: Gibt an, welche Experimentgruppe gestartet wird.
- ```45```: Zufalls-Seed für die Auswahl von Quelle und Ziel.
- ```5```: Anzahl der zu startenden Experimente.
- ```100```: Anzahl der Knoten im Graphen.
- ```5```: Anzahl der Quellknoten.
- ```RANDOM```: Art des Fehlers (alternativ: ```CLUSTER```)

## Ergebnisse

Der Haupt-Branch dieses Repositories enthält vorberechnete Ergebnisdateien. Um eigene Experimente durchzuführen, kann der separaten clean-Branch verwendet werden. Dieser enthält eine saubere Umgebung ohne Resultate.

## Plot-Erzeugung

Alle Dateien, deren Name das Wort plot enthält, erzeugen die Diagramme, die in der finalen Version der Masterarbeit verwendet und referenziert wurden. Diese Skripte visualisieren experimentelle Ergebnisse zu Metriken wie Resilienz, Laufzeit, Pfadlänge (Hops) oder Strukturgröße.

Um die Plots zu erzeugen, genügt der Aufruf des jeweiligen Skripts:
```
python3 <plot_dateiname>.py
```
Es muss sichergestellt werden, dass die benötigten Ergebnisdateien im Verzeichnis results/ vorhanden sind. Dazu müssen ggf. Dateipfade oder Algorithmenbezeichner innerhalb der Skripte angepasst werden.
