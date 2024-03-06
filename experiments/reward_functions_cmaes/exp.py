import sys
import os
import shutil

from thesis_manta_ray.morphology.specification.default import default_morphology_specification
from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology
from thesis_manta_ray.controller.specification.default import default_controller_dragrace_specification 
from thesis_manta_ray.controller.specification.controller_specification import MantaRayCpgControllerSpecification
from thesis_manta_ray.controller.parameters import MantaRayControllerSpecificationParameterizer
from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG

from task.drag_race import Move, DragRaceTask
from fprs.specification import RobotSpecification
from cmaes import CMA

import numpy as np

# set to False if you want to start from scratch
continue_where_stopped = False



# copy the evolution_simulation.py file to this folder such that the visualizer can always be used after an experiment
src = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/evolution_simulation.py"
dest = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/experiments/reward_functions_cmaes/evolution_simulation_correct_version.py"

if not continue_where_stopped:
    shutil.copy2(src, dest)
    from evolution_simulation import OptimizerSimulation
else:
    from evolution_simulation_correct_version import OptimizerSimulation


for reward_fn in DragRaceTask.reward_functions:
    velocity = 0.7
    velocity_str = "{:.2f}".format(velocity)
    reward_str = reward_fn.replace("*", "_times_")
    file_path = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/experiments/reward_functions_cmaes/sim_objects/energy_velocity_" + velocity_str + "_reward_function_" + reward_str + ".pkl"
    print("file_path: ", file_path)
    if os.path.exists(file_path) and continue_where_stopped:
        print("skipping: ", file_path)
        continue

    # morphology
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)
    # parameterizer = MantaRayMorphologySpecificationParameterizer(
    #     torso_length_range=(0.05, 2.),
    #     torso_radius_range=(0.05, 2.),
    #     )
    # parameterizer.parameterize_specification(specification=morphology_specification)
    

    # controller
    simple_env = Move(velocity=velocity, reward_fn=reward_fn).environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), # TODO: remove this, ask Dries
                                                wrap2gym=False)
    observation_spec = simple_env.observation_spec()
    action_spec = simple_env.action_spec()
    names = action_spec.name.split('\t')
    index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
    index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
    controller_specification = default_controller_dragrace_specification(action_spec=action_spec)
    controller_parameterizer = MantaRayControllerSpecificationParameterizer(
        amplitude_fin_out_plane_range=(0, 1),
        frequency_fin_out_plane_range=(0, 1),
        offset_fin_out_plane_range=(0, np.pi),
    )
    controller_parameterizer.parameterize_specification(specification=controller_specification)
    cpg = CPG(specification=controller_specification,
            low=-1,
            high=1,
            )

    robot_spec = RobotSpecification(morphology_specification=morphology_specification,
                                    controller_specification=controller_specification)

    # morphology_space = parameterizer.get_target_parameters(specification=morphology_specification)
    bounds = np.zeros(shape=(len(controller_parameterizer.get_parameter_labels()), 2))
    bounds[:, 1] = 1
    cma = CMA(mean=np.random.uniform(low=0,
                                    high=1,
                                    size=len(controller_parameterizer.get_parameter_labels())),
            sigma=0.05,
            bounds=bounds,
            population_size=10,    # has to be more than 1
            lr_adapt=True,
            )
    sim = OptimizerSimulation(
        task_config=Move(simulation_time=10, velocity=velocity, reward_fn=reward_fn),
        robot_specification=robot_spec,
        parameterizer=controller_parameterizer,
        population_size=10,  # make sure this is a multiple of num_envs
        num_generations=200,
        outer_optimalization=cma,
        controller=CPG,
        skip_inner_optimalization=True,
        record_actions=True,
        action_spec=action_spec,
        num_envs=10,
        logging=True,
        )
    print("sim, vel: ", velocity, "reward_fn: ", reward_fn, ", done.")

    sim.run()
    # best_gen, best_episode = sim.get_best_individual()
    sim.finish(store=True, name="reward_functions_cmaes/sim_objects/energy_velocity_"+velocity_str+"_reward_function_"+reward_str)