import sys
import os

import numpy as np

from task.drag_race import DragRaceTask

sys.path.append(os.path.abspath("/media/ruben/data/documents/unief/thesis"))

import matplotlib.pyplot as plt
from thesis_manta_ray.evolution_simulation import OptimizerSimulation

# Load the .pkl file
file_path = "/media/ruben/data/documents/unief/thesis/results_and_visualization/simulation_objects/2024-02-02_12_00_13.pkl"

for reward_fn in DragRaceTask.reward_functions:
    for velocity in list(np.linspace(0.2, 2., 7)):
        try:
            sim = OptimizerSimulation.load("/media/ruben/data/documents/unief/thesis/thesis_manta_ray/results_and_visualization/simulation_objects/test_drag_race.pkl")
        except Exception as e:
            print(e)
            print("Could not load the .pkl file, make sure that the class has not changed since the last save")
        sim.visualize()


