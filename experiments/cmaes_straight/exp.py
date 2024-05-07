import sys
import os
import shutil

from thesis_manta_ray.morphology.specification.default import default_morphology_specification
from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology
from thesis_manta_ray.controller.specification.default import default_controller_specification 
from thesis_manta_ray.controller.specification.controller_specification import MantaRayCpgControllerSpecification
from thesis_manta_ray.controller.parameters import MantaRayControllerSpecificationParameterizer
from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG

from thesis_manta_ray.task.drag_race import MoveConfig, Task
from fprs.specification import RobotSpecification
from cmaes import CMA

import numpy as np

# set to False if you want to reload the latest version of the simulation and start from scratch
continue_where_stopped = False



# copy the evolution_simulation.py file to this folder such that the visualizer can always be used after an experiment
src = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/evolution_simulation.py"
dest = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/experiments/cmaes_straight/evolution_simulation_correct_version.py"

if not continue_where_stopped:
    shutil.copy2(src, dest)
    from evolution_simulation import OptimizerSimulation
else:
    from evolution_simulation_correct_version import OptimizerSimulation


if __name__ == "__main__":
    # morphology
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)
    # parameterizer = MantaRayMorphologySpecificationParameterizer(
    #     torso_length_range=(0.05, 2.),
    #     torso_radius_range=(0.05, 2.),
    #     )
    # parameterizer.parameterize_specification(specification=morphology_specification)
    

    # task
    config = MoveConfig(simulation_time=10, 
                         velocity=0.5,
                         reward_fn="Δx_random_target",
                         task_mode="random_target",)
    config.target_location = np.array([-config.velocity*config.simulation_time, 0, 1.5])

    # controller
    simple_env = config.environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), 
                                                wrap2gym=False)
    observation_spec = simple_env.observation_spec()
    action_spec = simple_env.action_spec()
    names = action_spec.name.split('\t')
    index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
    index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
    controller_specification = default_controller_specification(action_spec=action_spec)
    controller_parameterizer = MantaRayControllerSpecificationParameterizer()
    controller_parameterizer.parameterize_specification(specification=controller_specification)
    print(f"controller: {controller_specification}")
    cpg = CPG(specification=controller_specification,
              low=-1,
              high=1,
              )

    robot_spec = RobotSpecification(morphology_specification=morphology_specification,
                                    controller_specification=controller_specification)

    # morphology_space = parameterizer.get_target_parameters(specification=morphology_specification)
    bounds = np.zeros(shape=(len(controller_parameterizer.get_parameter_labels()), 2))    # minus 1 for the phase bias
    bounds[:, 1] = 1
    cma = CMA(mean=np.random.uniform(low=0,
                                     high=1,
                                     size=len(controller_parameterizer.get_parameter_labels())),
              sigma=0.05,
              bounds=bounds,
              population_size=10,    # has to be more than 1
              lr_adapt=True,
              seed=42
              )

    sim = OptimizerSimulation(
        task_config=config,
        robot_specification=robot_spec,
        parameterizer=controller_parameterizer,
        population_size=10,  # make sure this is a multiple of num_envs
        num_generations=2,
        outer_optimalization=cma,
        controller=CPG,
        skip_inner_optimalization=True,
        record_actions=True,
        action_spec=action_spec,
        num_envs=10,
        logging=False,
        )
    
    sim.run()

    # sim.plot_observations(normalised_action=sol.parameters,
    #                       observation_name="task/avg_angular_velocity")
    # sim.plot_observations(normalised_action=sol.parameters,
    #                         observation_name="task/angular_velocity")
    # sim.plot_observations(normalised_action=sol.parameters,
    #                         observation_name="task/orientation")
    # sim.viewer(normalised_action=sol.parameters)


    parameters = sim.get_best_individual(action=True)
    sim.viewer(parameters)
    sim.visualize()
    # sim.viewer_gen_episode(generation=best_gen, episode=best_episode)