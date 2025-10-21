import numpy as np
from controller.cmaes_cpg_vectorized import CPG
from controller.parameters import MantaRayControllerSpecificationParameterizer
from controller.specification.default import default_controller_specification
from evolution_simulation import OptimizerSimulation
from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.default import default_morphology_specification
from task.drag_race import Task, MoveConfig
from controller.quality_diversity import Archive, MapElites
from cmaes import CMA
from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG
from fprs.specification import RobotSpecification
from scipy.spatial.transform import Rotation

archive = Archive.load("experiments/qd_v0.5_differential/sim_objects/archive.pkl")
print(f"archive parameter names: {archive._parameter_names}")
print(f"number of solutions: {len(archive.solutions)}")
velocities = []
for sol in archive:
    try:
        velocities.append(sol.metadata['avg_velocity'])
    except KeyError:
        continue
print(f"number of velocities: {len(velocities)}")
print(f"average velocity: {np.mean(velocities)}")
# print(f"number of bins: {len(archive.bins)}")

# plot the distance between neighbours
# archive.plot_distance_neighbours_distribution(parameter_names=["fin_offset_left"],
#                                               title="Difference of left fin offset between neighbours",
#                                               filename="experiments/qd_v0.5_differential/plots/distance_neighbours_distribution_fin_offset_left",
#                                               show=True,
#                                               )
# archive.plot_distance_neighbours_distribution(parameter_names=['fin_amplitude_left', 'fin_offset_left', 'frequency_left', 'phase_bias_left', 'fin_amplitude_right', 'fin_offset_right', 'frequency_right', 'phase_bias_right'],
#                                               title="Distance between neighbour's parameters",
#                                               filename="experiments/qd_v0.5_differential/plots/distance_neighbours_distribution",
#                                               show=True,
#                                               print_above=0.925,
#                                               )
# plot the fitness values from the archive in a histogram with plotly
archive.plot_fitness_distribution()