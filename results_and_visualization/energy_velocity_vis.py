import sys
import os

sys.path.append(os.path.abspath("/media/ruben/data/documents/unief/thesis"))

import matplotlib.pyplot as plt
from thesis_manta_ray.evolution_simulation import OptimizerSimulation

# Load the .pkl file
file_path = "/media/ruben/data/documents/unief/thesis/results_and_visualization/simulation_objects/2024-02-02_12_00_13.pkl"

try:
    sim = OptimizerSimulation.load("/media/ruben/data/documents/unief/thesis/thesis_manta_ray/results_and_visualization/simulation_objects/test.pkl")
except Exception as e:
    print(e)
    print("Could not load the .pkl file, make sure that the class has not changed since the last save")
    
sim.visualize()


