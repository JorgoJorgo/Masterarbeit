#!/bin/bash

# Cluster-Experimente
python3 planar_experimentsRANDOM_ZOO_57.py zoo 123 1 20 18 CLUSTER &
python3 planar_experimentsRANDOM_ZOO_102.py zoo 123 1 20 18 CLUSTER &
python3 planar_experimentsRANDOM_ZOO_136.py zoo 123 1 20 18 CLUSTER &


# Random-Experimente
python3 planar_experimentsRANDOM_ZOO_57.py zoo 123 1 20 18 RANDOM &
python3 planar_experimentsRANDOM_ZOO_102.py zoo 123 1 20 18 RANDOM &
python3 planar_experimentsRANDOM_ZOO_136.py zoo 123 1 20 18 RANDOM &

# Auf alle Prozesse warten
wait
