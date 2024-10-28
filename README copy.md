In this repository, you will find the code for the Studienarbeit "Analyzing Network Routing Resilience: A Hybrid Approach of Face and Tree Routing" by Georgios Karamoussanlis. For detailed information, please refer to the readme file from the original [fast failover](https://gitlab.cs.univie.ac.at/ct-papers/fast-failover) framework on which this repository is based.


## Requirements

This repository has been tested with Ubuntu 22.04. Additional required modules can be installed with:
```
pip install networkx numpy matplotlib pydot
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
python3 one_tree_with_checkpoint_experiments.py regular 0 5 50 5
```
Explanation of the inputs (from left to right):

- ```regular``` : Specifies which experiments to run with the same parameters.
- ```0``` : Random seed for choosing the source and destination.
- ```5``` : Number of experiments to run.
- ```100``` : Number of nodes in the graph.
- ```5``` : Number of sources to route a packet to the destination.

The unchanged regular algorithm then creates random graphs with ```n``` Nodes and a connectivity of 5 (```k```). <br />
By changing the limits of the ```for i in range(1,13):```in the main function the user is able to change the limit of the inserted fails. <br />
In each run of the ```for-loop``` 5 (```f_num```) edges get added to the failed links.
## Running Clustered Failures on Real World Graphs

To start the experiments with graphs from the Topology Zoo using clustered failures, execute the following command:

```
python3 new_clustered_experiments.py zoo 45 5 100 5 40 CLUSTER False
```
Explanation of the inputs (from left to right):

- ```zoo``` : Specifies which experiments to run with the same parameters.
- ```45``` : Random seed for choosing the source and destination.
- ```5``` : Number of experiments to run.
- ```100``` : Number of nodes in the graph.
- ```5``` : Number of sources to route a packet to the destination.
- ```40``` : Number of failed links.
- ```CLUSTER``` : Method for choosing edge failures.
- ```False``` : If true, only small zoo graphs with fewer than 25 nodes are run.
