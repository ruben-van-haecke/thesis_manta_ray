import sys
import os

sys.path.append(os.path.abspath("/media/ruben/data/documents/unief/thesis"))

from evolution_simulation import EvolutionSimulation
from morphology.specification.default import default_morphology_specification
from morphology.morphology import MJCMantaRayMorphology
from controller.specification.default import default_controller_dragrace_specification 
from controller.specification.controller_specification import MantaRayCpgControllerSpecification
from controller.parameters import MantaRayControllerSpecificationParameterizer

from task.drag_race import Move
from fprs.specification import RobotSpecification
from evolution_simulation import OptimizerSimulation
from cmaes import CMA
from controller.cmaes_cpg_vectorized import CPG

import numpy as np


for velocity in range(0, 2, 0.1):
    # morphology
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)
    # parameterizer = MantaRayMorphologySpecificationParameterizer(
    #     torso_length_range=(0.05, 2.),
    #     torso_radius_range=(0.05, 2.),
    #     )
    # parameterizer.parameterize_specification(specification=morphology_specification)
    

    # controller
    simple_env = Move().environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), # TODO: remove this, ask Dries
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
              sigma=0.005,
              bounds=bounds,
              population_size=10,    # has to be more than 1
              lr_adapt=True,
              )
    sim = OptimizerSimulation(
        task_config=Move(simulation_time=10, velocity=velocity),
        robot_specification=robot_spec,
        parameterizer=controller_parameterizer,
        population_size=10,  # make sure this is a multiple of num_envs
        num_generations=1,
        outer_optimalization=cma,
        controller=CPG,
        skip_inner_optimalization=True,
        record_actions=True,
        action_spec=action_spec,
        num_envs=10,
        logging=True,
        )
    
    sim.run()
    best_gen, best_episode = sim.get_best_individual()
    sim.finish(store=True, name="energy_velocity_"+str(i))