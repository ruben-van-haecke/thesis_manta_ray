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
# print(f"number of bins: {len(archive.bins)}")

# plot the distance between neighbours
archive.plot_distance_neighbours_distribution(parameter_name="fin_amplitude_left",
                                              title="Difference in left fin amplitude between neighbours",
                                              filename="experiments/qd_v0.5_differential/plots/distance_neighbours_distribution_fin_amplitude_left",
                                              show=True,
                                              )
