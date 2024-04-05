import numpy as np
from controller.cmaes_cpg_vectorized import CPG
from controller.parameters import MantaRayControllerSpecificationParameterizer
from controller.specification.default import default_controller_dragrace_specification
from evolution_simulation import OptimizerSimulation
from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.default import default_morphology_specification
from task.drag_race import Task, MoveConfig
from controller.quality_diversity import Archive, MapElites
from cmaes import CMA
from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG
from fprs.specification import RobotSpecification
from scipy.spatial.transform import Rotation


 # morphology
morphology_specification = default_morphology_specification()
morphology = MJCMantaRayMorphology(specification=morphology_specification)

# controller
simple_env = MoveConfig().environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), # TODO: remove this, ask Dries
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
            seed=42
            )

archive: Archive = Archive.load("experiments/qd_v0.5_differential/sim_objects/archive.pkl")
map_elites = MapElites(archive)
simulation_time = 10
velocity = 0.5
config = MoveConfig(simulation_time=simulation_time, 
                        velocity=velocity,
                        reward_fn="(E + 200*Δx) * (Δx)",
                        task_mode="random_target")


if True:    # verify the point
    sol = archive.solutions[(4, 3, 5)][0]
    pitch, yawn = sol.behaviour[[1, 2]]
    parameters = sol.parameters
else:   # try a chosen point
    pitch = 0
    yawn = np.pi/4
    parameters = archive.interpolate(np.array([0, pitch, yawn]), k=5)

# pitch = np.pi/4
# yawn = np.pi/4
target_location = np.array([1, 0, 0])*simulation_time*velocity
# yawn
# target_location[:2] = np.array([np.cos(yawn), np.sin(yawn), -np.sin(yawn), np.cos(yawn)]).reshape((2, 2)) @ target_location[:2]
# # pitch
# target_location[[0, 2]] = np.array([np.cos(pitch), np.sin(pitch), -np.sin(pitch), np.cos(pitch)]).reshape(2, 2) @ target_location[[0, 2]]
# config.target_location = target_location + config.initial_position
r = Rotation.from_euler('xyz', [0, -pitch, yawn+np.pi])
config.target_location = r.apply(target_location) + config.initial_position


sim = OptimizerSimulation(
    task_config=config,
    robot_specification=robot_spec,
    parameterizer=controller_parameterizer,
    population_size=10,  # make sure this is a multiple of num_envs
    num_generations=1,
    outer_optimalization=map_elites,#cma,
    controller=CPG,
    skip_inner_optimalization=True,
    record_actions=True,
    action_spec=action_spec,
    num_envs=10,
    logging=False,
    )
# a.plot_grid("pitch", "roll")
# a.plot_grid("yawn", "pitch")
# a.plot_grid_3d("yawn", "pitch", "roll")
# parameters = a.interpolate(np.array([0, pitch, yawn]), k=10)
# print(f"parameters: {parameters}")
# sim.viewer(parameters)
sim.viewer(parameters)