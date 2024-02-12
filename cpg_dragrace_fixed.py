import numpy as np

from controller.cmaes_cpg_vectorized import CPG
from controller.specification.controller_specification import MantaRayCpgControllerSpecification
from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.default import default_morphology_specification
from task.drag_race import Move

from dm_control import viewer
from dm_env import TimeStep
import time as time_module
import matplotlib.pyplot as plt

if __name__ == "__main__":
    morpholoby_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morpholoby_specification)

    task_config = Move(simulation_time=20)
    dm_env = task_config.environment(morphology=manta_ray, wrap2gym=False)

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()
    controller_spec = MantaRayCpgControllerSpecification(num_neurons=8, action_spec=action_spec)

    # set the control_parameters of the oscillators
    tail_segment_0_z = 0
    tail_segment_0_y = 1
    tail_segment_1_z = 2
    tail_segment_1_y = 3
    right_fin_x = 4
    right_fin_z = 5
    left_fin_x = 6
    left_fin_z = 7
    # controller_spec.x.add_connections(connections=[(0, tail_segment_0_y), (0, tail_segment_1_y), (0, right_fin_x), (0, left_fin_x)],
    #                                   weights=[0., 0., 0., 0])
    omega = 2
    bias = np.pi
    controller_spec.r.set_connections(connections=[(0, 6), (0, 4)],
                                        weights=[1, 1.])
    controller_spec.omega.set_connections(connections=[(0, 6), (0, 4)],
                                        weights=np.ones(shape=(2, ))*np.pi*2*omega)
    connections = [#(1, 3), (3, 1), # connection between the two out of plane oscillators of the tail
            #    (3, 4), (4, 3),  # connection tail-rigth fin
            #    (3, 6), (6, 3),  # connection tail-left fin
                (6, 4), (4, 6),], # connection right-left fin
    controller_spec.weights.set_connections(connections=connections,
                                            weights=[4, 4.],)
    controller_spec.phase_biases.set_connections(connections=connections,
                                            weights=[-bias, bias])
    cpg = CPG(specification=controller_spec)
    timesteps_to_control = 1
    iterator = 0
    all_actions = np.empty(shape=(8, int(task_config.simulation_time/task_config.physics_timestep)))
    all_actions = cpg.ask(observation=dm_env.reset().observation,
                            duration=task_config.simulation_time,
                            sampling_period=task_config.physics_timestep)
    time = np.arange(0, task_config.simulation_time, task_config.physics_timestep)
    plt.plot(time, all_actions[4, :], label="left fin")
    plt.plot(time, all_actions[6, :], label="right fin")
    plt.legend()
    plt.show()

    # print(cpg)

    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec, cpg, iterator, timesteps_to_control
        time = timestep.observation["task/time"][0]
        if time == 0:
            iterator = 0
            iterator = 0

        num_actuators = action_spec.shape[0]
        actions = np.zeros(num_actuators)
        actions[4] = all_actions[4, iterator].flatten()
        actions[6] = all_actions[6, iterator].flatten()
        iterator = (iterator + 1) % all_actions.shape[1]

        # rescale from [-1, 1] to actual joint range
        minimum, maximum = action_spec.minimum, action_spec.maximum

        normalised_actions = (actions + 1) / 2

        scaled_actions = minimum + normalised_actions * (maximum - minimum)
        # print("action: ", scaled_actions)
        return scaled_actions


    viewer.launch(
            environment_loader=dm_env, 
            policy=oscillator_policy_fn
            )