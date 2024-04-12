# controller
from task.bezier_parkour import BezierParkour
from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG
from thesis_manta_ray.controller.parameters import MantaRayControllerSpecificationParameterizer
from thesis_manta_ray.controller.specification.default import default_controller_specification
from thesis_manta_ray.controller.quality_diversity import Archive
from thesis_manta_ray.controller.rule_based import RuleBased

from thesis_manta_ray.evolution_simulation import OptimizerSimulation
from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology
from thesis_manta_ray.morphology.specification.default import default_morphology_specification
from thesis_manta_ray.task.drag_race import Task, MoveConfig
from cmaes import CMA
from fprs.specification import RobotSpecification
from scipy.spatial.transform import Rotation
from dm_env import TimeStep
from dm_control import viewer
import numpy as np


 # morphology
morphology_specification = default_morphology_specification()
morphology = MJCMantaRayMorphology(specification=morphology_specification)

# task and controller
simulation_time = 10
velocity = 0.5
parkour = BezierParkour.load("task/parkours/slight_curve.pkl")
config = MoveConfig(control_substeps=4,
                    simulation_time=simulation_time, 
                    velocity=velocity,
                    reward_fn="(E + 200*Δx) * (Δx)",
                    task_mode="parkour",
                    parkour=parkour)
dm_env = config.environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), wrap2gym=False)
observation_spec = dm_env.observation_spec()
action_spec = dm_env.action_spec()
names = action_spec.name.split('\t')
index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')

# controller
controller_specification = default_controller_specification(action_spec=action_spec)
controller_parameterizer = MantaRayControllerSpecificationParameterizer(
)
controller_parameterizer.parameterize_specification(specification=controller_specification)
cpg = CPG(specification=controller_specification,
            low=-1,
            high=1,
            )

robot_spec = RobotSpecification(morphology_specification=morphology_specification,
                                controller_specification=controller_specification)


archive: Archive = Archive.load("experiments/qd_v0.5_differential/sim_objects/archive.pkl")
rule_based_layer = RuleBased(archive=archive)


minimum, maximum = action_spec.minimum.reshape(-1, 1), action_spec.maximum.reshape(-1, 1)   # shapes (n_neurons, 1)

def policy(timestep: TimeStep) -> np.ndarray:
    time = timestep.observation["task/time"][0]
    obs = timestep.observation
    # update the controller modulation
    scaled_action = rule_based_layer.select_parameters(current_angular_positions=obs["task/orientation"][0],
                                                       current_xyz_velocities=obs["task/xyz_velocity"][0],
                                                       current_position=obs["task/position"][0],
                                                       parkour=parkour)
    # if time < 5:
    #     scaled_action = archive.solutions[(6, 11, 3)][0].parameters
    # else:
    #     scaled_action = archive.solutions[(6, 1, 3)][0].parameters
    # controller_parameterizer.parameter_space(specification=controller_specification,
    #                                          controller_action=scaled_action,)

    # actuation
    normalised_action = (cpg.ask(observation=timestep.observation,
                                    duration=None,  # one time step
                                    sampling_period=config.physics_timestep
                                    )+1)/2
    scaled_action = minimum + normalised_action * (maximum - minimum)
    return scaled_action[:, 0]
viewer.launch(
    environment_loader=dm_env, 
    policy=policy
    )