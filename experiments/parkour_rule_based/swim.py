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
import matplotlib.pyplot as plt

import plotly.graph_objects as go


 # morphology
morphology_specification = default_morphology_specification()
morphology = MJCMantaRayMorphology(specification=morphology_specification)

# task and controller
simulation_time = 40
velocity = 0.5
parkour = BezierParkour.load("task/parkours/slight_curve.pkl")
# config = MoveConfig(control_substeps=20,
#                     simulation_time=simulation_time, 
#                     velocity=velocity,
#                     reward_fn="(E + 200*Δx) * (Δx)",
#                     task_mode="parkour",
#                     parkour=parkour)
config = MoveConfig(control_substeps=1,
                    simulation_time=simulation_time, 
                    velocity=velocity,
                    reward_fn="(E + 200*Δx) * (Δx)",
                    task_mode="random_target",
                    )
print(f"control_timestep: {config.control_timestep}")
config.target_location = np.array([-5, 2, 1.5])
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
left_actuation = []
right_actuation = []
phase_bias = [] # indexes 3 and 7
behaviour_previous = None
parameters_controller_previous = None
difference_behaviour = []
difference_parameters_controller = []

control_step = 1.  # time between cpg modulations
scaled_actions = np.empty((8, int(control_step/config.physics_timestep)))
counter = 0

def policy(timestep: TimeStep) -> np.ndarray:
    global left, right, phase_bias, behaviour_previous, parameters_controller_previous, config, scaled_actions, counter, control_step
    time = timestep.observation["task/time"][0]
    if np.allclose(time[0] % control_step,  0., atol=0.01):
        print("changing parameters")
        obs = timestep.observation
        # update the controller modulation
        if config.task_mode == "parkour":
            scaled_action, behaviour_descriptor = rule_based_layer.select_parameters_parkour(current_angular_positions=obs["task/orientation"][0],
                                                            current_xyz_velocities=obs["task/xyz_velocity"][0],
                                                            current_position=obs["task/position"][0],
                                                            parkour=parkour)
        elif config.task_mode == "random_target":
            scaled_action, behaviour_descriptor = rule_based_layer.select_parameters_target(current_angular_positions=obs["task/orientation"][0],
                                                            current_xyz_velocities=obs["task/xyz_velocity"][0],
                                                            current_position=obs["task/position"][0],
                                                            target_location=config.target_location,
                                                            print_flag=True,
                                                            scaling=True)
        else:
            raise ValueError(f"task_mode: {config.task_mode} not supported")
        controller_parameterizer.parameter_space(specification=controller_specification,
                                                controller_action=scaled_action,)
        normalised_actions = (cpg.ask(observation=timestep.observation,
                                    duration=1.,  
                                    sampling_period=config.physics_timestep
                                    )+1)/2
        # scaling
        scaled_actions = minimum + normalised_actions * (maximum - minimum)
        phase_bias.append(scaled_action[3])
        if behaviour_previous is not None:
            difference_behaviour.append(np.linalg.norm(behaviour_previous - behaviour_descriptor))
        behaviour_previous = behaviour_descriptor
        if parameters_controller_previous is not None:
            difference_parameters_controller.append(np.linalg.norm(parameters_controller_previous - scaled_action))
        parameters_controller_previous = scaled_action
        counter = 0

    scaled_action = scaled_actions[:, counter]
    left_actuation.append(scaled_action[index_left_pectoral_fin_x])
    right_actuation.append(scaled_action[index_right_pectoral_fin_x])
    counter += 1
    return scaled_action


viewer.launch(
    environment_loader=dm_env, 
    policy=policy
    )




if False:
    time = np.linspace(0, simulation_time, len(left_actuation))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=left_actuation, name='Left fin'))
    fig.add_trace(go.Scatter(x=time, y=right_actuation, name='Right fin'))

    fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Actuation',
        legend=dict(font=dict(size=16)),  # Increase the font size of the legend
        xaxis=dict(title=dict(font=dict(size=16))),
        yaxis=dict(title=dict(font=dict(size=16)))
    )

    fig.show()

    time = np.linspace(0, simulation_time, len(phase_bias))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=phase_bias, name='Phase Bias'))
    fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Phase Bias',
        legend=dict(font=dict(size=16)),  # Increase the font size of the legend
        xaxis=dict(title=dict(font=dict(size=16))),
        yaxis=dict(title=dict(font=dict(size=16)))
    )
    fig.show()

    time = np.linspace(0, simulation_time, len(difference_behaviour))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=difference_behaviour, name='Difference Behaviour'))
    fig.add_trace(go.Scatter(x=time, y=difference_parameters_controller, name='Difference Parameters Controller'))
    fig.update_layout(
        xaxis_title='Time [s]',
        yaxis_title='Difference',
        legend=dict(font=dict(size=16)),  # Increase the font size of the legend
        xaxis=dict(title=dict(font=dict(size=16))),
        yaxis=dict(title=dict(font=dict(size=16)))
    )
    fig.show()

    print(f"pearson matrix: {np.corrcoef(difference_behaviour, difference_parameters_controller)}")
    print(f"pearson correlation: {np.corrcoef(difference_behaviour, difference_parameters_controller)[0, 1]}")