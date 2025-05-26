# Masterarbeit

In this repository, you will find the code for the Master Thesis "Untersuchung von Fehlerresilienz in planaren Netzwerken mit lokalen Routing-Regeln" by Georgios Karamoussanlis. For detailed information, please refer to the readme file from the original [fast failover](https://gitlab.cs.univie.ac.at/ct-papers/fast-failover) framework on which this repository is based.


## Requirements

This repository has been tested with Ubuntu 22.04. Additional required modules can be installed with:
```
pip install networkx==3.2 numpy matplotlib pydot
```


## Overview

* `trees.py`: Contains all the algorithms for tree formation and their helper functions.
* `trees_with_cp.py`: Contains all the algorithms for the tree formation using a checkpoint and the needed functions using faces
* `routing.py`: Contains the routing algorithms.
* `benchmark_graphs`: Folder for the topologies used.
* `results`: Folder for the outputs and results of the algorithms.
* `..._experiments.py`: Experiments with preset parameters ready for execution.
* The individual results are grouped into folders that include the benchmarks, experiments, and log files for each failure rate.
* `benchmark-....txt`: Available for each failure rate of an experiment. These files can be used in `plotter.py` by adjusting the file path and algorithm names to match the result file.

The topologies can be found at [Rocketfuel](https://research.cs.washington.edu/networking/rocketfuel/) and [Internet Topology Zoo](http://www.topology-zoo.org/). These need to be downloaded and placed in the `benchmark_graphs` folder.

## Running Random Failures on Random Regular Created Graphs

To start the experiments with random generated graphs, execute the following command:

```
python3 planar_experiments.py planar 3 1 45 15 RANDOM
```
Explanation of the inputs (from left to right):

- ```planar``` : Specifies which experiments to run with the same parameters.
- ```3``` : Random seed for choosing the source and destination.
- ```1``` : Number of experiments to run.
- ```45``` : Number of nodes in the graph.
- ```15``` : Number of sources to route a packet to the destination.
- ```RANDOM``` : Type of failures (else ```CLUSTER```)

The unchanged ```planar``` algorithm then creates random graphs with ```n``` Nodes and is trying to use Delaunay Triangulation or Gabriel-Graph-Method to make it planar. <br />
By changing the limits of the ```for i in range(1,13):```in the main function the user is able to change the limit of the inserted fails. <br />
In each run of the ```for-loop``` 5 (```f_num```) edges get added to the failed links.


## Real World Graphs

To start the experiments with graphs from the Topology Zoo using clustered failures, execute the following command:

```
python3 planar_experimentspy zoo 45 5 100 5 RANDOM
```
Explanation of the inputs (from left to right):

- ```zoo``` : Specifies which experiments to run with the same parameters.
- ```45``` : Random seed for choosing the source and destination.
- ```5``` : Number of experiments to run.
- ```100``` : Number of nodes in the graph.
- ```5``` : Number of sources to route a packet to the destination.
- ```RANDOM``` : Type of failures (else ```CLUSTER```)


## Results
The repository’s main branch contains pre-generated result files for reference.
For custom testing or experimenting with the algorithms without any precomputed results, you can use the separate clean branch, which provides a fresh setup without result data.


## Plotting

All files that contain the word `plot` in their name generate the exact plots used and referenced in the final version of the Master Thesis.  
These scripts visualize the results of the experiments based on various metrics such as resilience, runtime, number of hops, or structure size.

To reproduce the plots, simply run the corresponding script using:


```
python3 <plot_file_name>.py
```
Make sure the required result files are present in the `results/` directory, and adjust any file paths or algorithm names inside the script if needed.
