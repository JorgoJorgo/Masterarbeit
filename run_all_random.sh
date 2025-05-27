#!/bin/bash

# Cluster-Experimente
python3 planar_experimentsCLUSTER_DELAUNAY.py planar 123 1 45 20 CLUSTER & 
python3 planar_experimentsCLUSTER_GABRIEL.py planar 123 1 45 20 CLUSTER & 

# Random-Experimente
python3 planar_experimentsRANDOM_DELAUNAY.py planar 123 1 45 20 RANDOM & 
python3 planar_experimentsRANDOM_GABRIEL.py planar 123 1 45 20 RANDOM & 

wait
