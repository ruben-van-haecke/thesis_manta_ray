import pickle
from typing import List, Type

import gymnasium as gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import wandb
import copy
import time
import thesis_manta_ray
from datetime import datetime
from cmaes import CMA
from thesis_manta_ray.controller.quality_diversity import Solution, Archive, MapElites

from thesis_manta_ray.controller.cmaes_cpg_vectorized import CPG
from thesis_manta_ray.controller.parameters import MantaRayControllerSpecificationParameterizer
from thesis_manta_ray.controller.specification.controller_specification import MantaRayCpgControllerSpecification
from thesis_manta_ray.controller.specification.default import default_controller_dragrace_specification
from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology

from thesis_manta_ray.morphology.specification.default import default_morphology_specification
from thesis_manta_ray.parameters import MantaRayMorphologySpecificationParameterizer
from task.drag_race import MoveConfig
from fprs.specification import RobotSpecification

from mujoco_utils.environment import MJCEnvironmentConfig
from dm_control import viewer
from dm_env import TimeStep
from gymnasium.core import ObsType


warnings.filterwarnings('ignore', category=DeprecationWarning)


class OptimizerSimulation:
    """
    This class simulates the evolution of a population by means of an evolutionary strategy i.e. by distribution sampling
    of MantaRay agents. It is parallelized with CPU cores.
    It is responsible for:
    - running the simulation
    - saving the results
    """
    def __init__(self,
                task_config: MJCEnvironmentConfig,
                robot_specification: RobotSpecification,
                parameterizer: MantaRayControllerSpecificationParameterizer,
                population_size: int,
                num_generations: int,
                outer_optimalization ,
                controller: CPG,
                skip_inner_optimalization: bool = False,
                record_actions: bool = False,
                action_spec=None,
                num_envs: int = 2,
                logging: bool = True,
                ) -> None:
        """
        :param num_generations: the number of generations to run the simulation, if None keep running untill manually stopped
        :param outer_optimalization: the outer optimization algorithm responsible for 
        :param skip_inner_optimalization: whether to skip the inner optimization i.e. all actions are pre-computed to speed up the simulation
                                        this is a significant speed-up
        """
        assert population_size % num_envs == 0, "population_size should be a multiple of num_envs"
        self._task_config = task_config
        self._robot_specification = robot_specification
        self._parameterizer = parameterizer
        self._population_size = population_size
        self._num_generations = num_generations
        self._outer_optimalization = outer_optimalization
        self._controller: Type = controller
        self._skip_inner_optimalization = skip_inner_optimalization
        self._record_actions = record_actions
        self._num_envs = num_envs
        self._logging = logging

        self._controller_specs = [default_controller_dragrace_specification(action_spec=action_spec) for _ in range(self._num_envs)]
        for controller_spec in self._controller_specs:
            self._parameterizer.parameterize_specification(specification=controller_spec)
        self._morph_specs = [default_morphology_specification() for _ in range(self._num_envs)]
        time.sleep(5)
        self._controllers: List[CPG] = [self._controller(specification=controller_spec) for controller_spec in self._controller_specs]
        self._configs: List[MoveConfig] = [copy.deepcopy(task_config) for _ in range(self._num_envs)]

        succeed = False
        while not succeed:
            try:
                self._gym_env = gym.vector.AsyncVectorEnv([lambda: self._configs[env_id].environment(morphology=MJCMantaRayMorphology(specification=self._morph_specs[env_id]),
                                                                        wrap2gym=True) for env_id in range(self._num_envs)])
                succeed = True
            except:
                print("Failed to create the gym environment, retrying...")
                time.sleep(5)


        self._action_spec = action_spec
        self._morphology_specification = self._robot_specification.morphology_specification
        self._controller_specification = self._robot_specification.controller_specification
        self._action_space = self._gym_env.action_space
        self._outer_rewards = np.zeros(shape=(self._num_generations, self._population_size))
        if self._skip_inner_optimalization:
            self._inner_rewards = None
        else:
            self._inner_rewards = np.zeros(shape=(self._num_generations,
                                                self._population_size, 
                                                task_config.simulation_time/task_config.control_timestep))
        if self._record_actions:
            self._actions = np.zeros(shape=(self._num_generations, 
                                            self._population_size, 
                                            8,  # action space
                                            int(np.ceil(task_config.simulation_time/task_config.control_timestep))+1))
            self._control_actions = np.zeros(shape=(self._num_generations, 
                                                    self._population_size, 
                                                    len(self._parameterizer.get_parameter_labels()),))
        # record observations
        self._observations = np.zeros(shape=(self._num_generations, 
                                            self._population_size, 
                                            3,  # observation space
                                            int(np.ceil(task_config.simulation_time/task_config.control_timestep))+1))
        # logging
        if self._logging:
            wandb.init(project="ruben_van_haecke_thesis", 
                    name=f"""{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_test_logging""",
                        config={
                            "generations": self._num_generations,
                            "duration": self._task_config.simulation_time,
                            "control_timestep": self._task_config.control_timestep,
                            "population_size": self._population_size,
                        }
                    )
    
    def run_episode_single(self,
                           generation: int,
                           episode: int,
                           counter: int,
                           obs: ObsType,
                           env_id: int,
                           ):
        minimum, maximum = self._action_spec.minimum.reshape(-1, 1), self._action_spec.maximum.reshape(-1, 1)   # shapes (n_neurons, 1)
        if self._skip_inner_optimalization:
            normalised_action = (self._controllers[env_id].ask(observation=obs,
                                                               duration=self._task_config.simulation_time,
                                                               sampling_period=self._task_config.physics_timestep
                                                               )+1)/2
        else:
            normalised_action = (self._controllers[env_id].ask(observation=obs)+1)/2
        # assert np.all(normalised_action >= 0) and np.all(normalised_action <= 1), "Action is not in [0, 1]"

        # record actions
        scaled_action = minimum + normalised_action * (maximum - minimum)
        if self._record_actions and self._skip_inner_optimalization: 
            self._actions[generation, episode, :, :] = scaled_action

        elif self._record_actions and not self._skip_inner_optimalization:
            self._actions[generation, episode, :, counter] = scaled_action

    def run_episode_parallel(self,
                    generation: int,
                    episode: int,
                    ) -> List[float]:
        """
        returns the reward and observations of the episode
        """
        done = False
        obs, info = self._gym_env.reset()
        self.obs = obs
        counter = 0
        while not done:
            scaled_action = np.zeros(shape=(self._num_envs, 8))
            for env_id in range(self._num_envs):
                # scaled_action[env_id, :] = 
                if not self._skip_inner_optimalization:
                    self.run_episode_single(generation=generation,
                                                            episode=episode+env_id,
                                                            counter=counter,
                                                            obs=obs,
                                                            env_id=env_id,
                                                            )
                elif obs["task/time"][0][0] == 0:
                    self.run_episode_single(generation=generation,
                                                            episode=episode+env_id,
                                                            counter=counter,
                                                            obs=obs,
                                                            env_id=env_id,
                                                            )
            last_obs = obs  # needed because the last observations are all zeroes 
            obs, reward, terminated, truncated, info = self._gym_env.step(self._actions[generation, episode:episode+self._num_envs, :, counter])
            self._observations[generation, episode:episode+self._num_envs, :, counter] = obs['task/delta_orientation']
            done = np.all(np.logical_or(terminated, truncated))
            counter += 1
        return reward, last_obs


    def run_generation(self,
                       generation: int,
                       ) -> None:
        solutions = []
        for agent in range(0, self._population_size, self._num_envs):
            outer_action = [self._outer_optimalization.ask() for _ in range(self._num_envs)]
            for env_id in range(self._num_envs):
                self._parameterizer.parameter_space(specification=self._controller_specs[env_id],
                                                    controller_action=outer_action[env_id])
            reward, obs = self.run_episode_parallel(generation=generation, episode=agent)

            if isinstance(self._outer_optimalization, CMA):
                solutions += [(single_action, single_reward) for single_action, single_reward in zip(outer_action, reward)]
            elif isinstance(self._outer_optimalization, MapElites):
                for env_id in range(self._num_envs):
                    # sol = Solution(behaviour=obs['task/orientation'][env_id, :], 
                    #                           fitness=1/reward[env_id], # fitness has to be optimized
                    #                           parameters=outer_action[env_id])
                    sol = Solution(behaviour=obs['task/delta_orientation'][env_id, :], 
                                              fitness=1/reward[env_id], # fitness has to be optimized
                                              parameters=outer_action[env_id])
                    solutions.append(sol)
            else:
                raise NotImplementedError(f"This outer_optimalization is not implemented, type: {type(self._outer_optimalization)}")
            

            for env_id in range(self._num_envs):
                self._outer_rewards[generation, agent+env_id] = reward[env_id]
                self._control_actions[generation, agent+env_id, :] = outer_action[env_id]
        self._outer_optimalization.tell(solutions)
    

    def run(self):
        for gen in range(self._num_generations):
            self.run_generation(generation=gen)
            if self._logging:    wandb.log({"generation": gen,
                                    "average": np.mean(self._outer_rewards[gen]),
                                        "worst": np.max(self._outer_rewards[gen]),
                                        "best": np.min(self._outer_rewards[gen]),
                                        })
    
    def get_best_individual(self) -> tuple[int, int]:
        """
        returns (generation, episode) of the best individual in the population"""
        indices = np.unravel_index(np.argmin(self._outer_rewards, axis=None), self._outer_rewards.shape)
        print("Best individual (gen, episode): ", indices, " , reward: ", self._outer_rewards[indices], " , action: ", self._control_actions[indices])
        for index, value in enumerate(self._control_actions[indices]):
            print(self._parameterizer.get_parameter_labels()[index], ": ", value)
        return indices
    
    def visualize(self):
        plt.plot(np.mean(self._outer_rewards, axis=1), label="average")
        plt.plot(self._outer_rewards.max(axis=1), label="max")
        plt.plot(self._outer_rewards.min(axis=1), label="min")
        plt.xlabel("generation")
        plt.ylabel("reward")
        plt.legend()
        plt.show()
    
    def visualize_inner(self, generation: int, episode: int):
        if self._record_actions is False:
            print("Record actions was skipped")
            return
        plt.plot(np.linspace(0, self._task_config.simulation_time, len(self._actions[generation, episode, index_left_pectoral_fin_x, :])), 
                 self._actions[generation, episode, index_left_pectoral_fin_x, :], 
                 label="left fin")
        plt.plot(np.linspace(0, self._task_config.simulation_time, len(self._actions[generation, episode, index_left_pectoral_fin_x, :])),
                 self._actions[generation, episode, index_right_pectoral_fin_x, :], 
                 label="right fin")
        plt.xlabel("time [seconds]")
        plt.ylabel("output")
        plt.legend()
        plt.show()

    def viewer_gen_episode(self,
               generation: int,
               episode: int,
               ) -> None:
        assert self._record_actions, "Cannot visualize actions if they are not recorded"
        dm_env = self._task_config.environment(morphology=MJCMantaRayMorphology(specification=self._morphology_specification), wrap2gym=False)
        def policy(timestep: TimeStep) -> np.ndarray:
            time = timestep.observation["task/time"][0]
            action = np.zeros(shape=self._action_space.shape[0])
            action: np.ndarray = self._actions[generation, episode, :, int(time/self._task_config.control_timestep)]
            return action
        viewer.launch(
            environment_loader=dm_env, 
            policy=policy
            )
    
    def viewer(self,
               normalised_action: np.ndarray,
               ) -> None:
        assert self._record_actions, "Cannot visualize actions if they are not recorded"
        dm_env = self._task_config.environment(morphology=MJCMantaRayMorphology(specification=self._morphology_specification), wrap2gym=False)
        controller_spec = default_controller_dragrace_specification(action_spec=self._action_spec)
        self._parameterizer.parameterize_specification(specification=controller_spec)
        controller = self._controller(specification=controller_spec)
        self._parameterizer.parameter_space(specification=controller_spec,
                                                    controller_action=normalised_action)

        minimum, maximum = self._action_spec.minimum.reshape(-1, 1), self._action_spec.maximum.reshape(-1, 1)   # shapes (n_neurons, 1)
        normalised_action = (controller.ask(observation=None,
                                            duration=self._task_config.simulation_time,
                                            sampling_period=self._task_config.physics_timestep
                                            )+1)/2
        scaled_action = minimum + normalised_action * (maximum - minimum)

        def policy(timestep: TimeStep) -> np.ndarray:
            time = timestep.observation["task/time"][0]
            return scaled_action[:, int(time/self._task_config.control_timestep)]
        viewer.launch(
            environment_loader=dm_env, 
            policy=policy
            )
    
    def plot_actions(self, 
                     normalised_controller_action: np.ndarray,
                     ) -> None:
        dm_env = self._task_config.environment(morphology=MJCMantaRayMorphology(specification=self._morphology_specification), wrap2gym=False)
        controller_spec = default_controller_dragrace_specification(action_spec=self._action_spec)
        self._parameterizer.parameterize_specification(specification=controller_spec)
        controller = self._controller(specification=controller_spec)
        self._parameterizer.parameter_space(specification=controller_spec,
                                                    controller_action=normalised_controller_action)

        minimum, maximum = self._action_spec.minimum.reshape(-1, 1), self._action_spec.maximum.reshape(-1, 1)   # shapes (n_neurons, 1)
        normalised_action = (controller.ask(observation=self.obs,
                                            duration=self._task_config.simulation_time,
                                            sampling_period=self._task_config.physics_timestep
                                            )+1)/2
        scaled_action = minimum + normalised_action * (maximum - minimum)
        plt.plot(scaled_action[4, :], label="left fin", color="blue")
        plt.plot(scaled_action[6, :], label="right fin", color="orange")
        plt.xlabel("time [seconds]")
        plt.ylabel("output")
        plt.legend()
        plt.show()
    
    def plot_observations(self, 
                     normalised_action: np.ndarray,
                     ) -> None:
        dm_env = self._task_config.environment(morphology=MJCMantaRayMorphology(specification=self._morphology_specification), wrap2gym=False)
        controller_spec = default_controller_dragrace_specification(action_spec=self._action_spec)
        self._parameterizer.parameterize_specification(specification=controller_spec)
        controller = self._controller(specification=controller_spec)
        self._parameterizer.parameter_space(specification=controller_spec,
                                                    controller_action=normalised_action)

        minimum, maximum = self._action_spec.minimum.reshape(-1, 1), self._action_spec.maximum.reshape(-1, 1)   # shapes (n_neurons, 1)
        normalised_action = (controller.ask(observation=None,
                                            duration=self._task_config.simulation_time,
                                            sampling_period=self._task_config.physics_timestep
                                            )+1)/2
        scaled_action = minimum + normalised_action * (maximum - minimum)
        observations = np.zeros(shape=(3, int(np.ceil(self._task_config.simulation_time/self._task_config.control_timestep))+1))

        def policy(timestep: TimeStep) -> np.ndarray:
            time = timestep.observation["task/time"][0]
            observations[:, int(time/self._task_config.control_timestep)] = timestep.observation["task/delta_orientation"][0]
            return scaled_action[:, int(time/self._task_config.control_timestep)]
        viewer.launch(
            environment_loader=dm_env, 
            policy=policy
            )
        t = np.linspace(0, self._task_config.simulation_time, len(observations[0]))
        plt.plot(t, observations[0], label="roll")
        plt.plot(t, np.ones_like(t)*np.average(observations[0]), label="average roll")
        plt.plot(t, observations[1], label="pitch")
        plt.plot(t, np.ones_like(t)*np.average(observations[1]), label="average pitch")
        plt.plot(t, observations[2], label="yawn")
        plt.plot(t, np.ones_like(t)*np.average(observations[2]), label="average yawn")
        plt.xlabel("time [seconds]")
        plt.ylabel("output")
        plt.legend()
        plt.show()

    def finish(self, store=True, name=None):
        """
        args:
            store: whether to store self in a file
            name: name of the file to store the results in, relative from the experiments folder, 
                if None it is the date
        """
        if self._logging:
            wandb.finish()

        if store == True:
            if name is None:
                name = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
            path = f"experiments/{name}.pkl"
            with open(path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Stored the simulation object in {path}")
    
    @staticmethod
    def load(
            path: str
            ) -> 'OptimizerSimulation': # forward referencing
        """
        args: 
            path: relative from /experiments/
        """
        path = f"experiments/{path}.pkl"
        with open(path, 'rb') as handle:
            return pickle.load(handle)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_gym_env']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore the unpicklable entries.
        self._gym_env = gym.vector.AsyncVectorEnv([lambda: MoveConfig().environment(morphology=MJCMantaRayMorphology(specification=self._morph_specs[env_id]),
                                                                  wrap2gym=True) for env_id in range(self._num_envs)])
    

if __name__ == "__main__":
    # morphology
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)
    # parameterizer = MantaRayMorphologySpecificationParameterizer(
    #     torso_length_range=(0.05, 2.),
    #     torso_radius_range=(0.05, 2.),
    #     )
    # parameterizer.parameterize_specification(specification=morphology_specification)
    

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
    # cma = CMA(mean=np.random.uniform(low=0,
    #                                  high=1,
    #                                  size=len(controller_parameterizer.get_parameter_labels())),
    #           sigma=0.05,
    #           bounds=bounds,
    #           population_size=10,    # has to be more than 1
    #           lr_adapt=True,
    #           seed=42
    #           )
    denomenator = 2
    # parameters: ['fin_amplitude_left', 'fin_offset_left', 'frequency_left', 'phase_bias_left', 'fin_amplitude_right', 'fin_offset_right', 'frequency_right', 'phase_bias_right']
    archive = Archive(parameter_bounds=[(0, 1) for _ in range(len(controller_parameterizer.get_parameter_labels()))],
                      feature_bounds=[(-np.pi/denomenator, np.pi/denomenator), (-np.pi/2/denomenator, np.pi/2/denomenator), (-np.pi/denomenator, np.pi/denomenator)], 
                      resolutions=[8, 8, 8],
                      parameter_names=controller_parameterizer.get_parameter_labels(), 
                      feature_names=["roll", "pitch", "yawn"],
                      symmetry = [('phase_bias_right', 'phase_bias_left'), 
                                ('frequency_right', 'frequency_left'), 
                                ('fin_offset_right', 'fin_offset_left'), 
                                ('fin_amplitude_right', 'fin_amplitude_left'),
                                ],
                        max_items_per_bin=1
                      )
    map_elites = MapElites(archive)

    sim = OptimizerSimulation(
        task_config=MoveConfig(simulation_time=10, 
                         velocity=0.5,
                         reward_fn="(E + 200*Δx) * (Δx)",
                         task_mode="no_target",),
        robot_specification=robot_spec,
        parameterizer=controller_parameterizer,
        population_size=10,  # make sure this is a multiple of num_envs
        num_generations=4,
        outer_optimalization=map_elites,#cma,
        controller=CPG,
        skip_inner_optimalization=True,
        record_actions=True,
        action_spec=action_spec,
        num_envs=10,
        logging=False,
        )
    
    sim.run()
    # best_gen, best_episode = sim.get_best_individual()
    # # sim.visualize()
    # sim.viewer_gen_episode(generation=best_gen, episode=best_episode)
    map_elites.optimization_info()
    archive.plot_grid_3d(x_label="roll", y_label="pitch", z_label="yawn")
    # best_sol_first_bin = archive.get_best_solution(index=(0, 0, 0))
    # first_solution = next(iter(archive))
    # other_sol = archive.get_symmetric_solution(best_sol_first_bin)
    # sim.viewer(normalised_action=best_sol_first_bin.parameters)
    # sim.plot_actions(normalised_action=first_solution.parameters)
#     action = np.array([0.9736032, 0.75782657, 0.25533115, 0.04304449, 0.95805741, 0.73478035,
#  0.73896048, 0.504163  ])
#     sim.plot_actions(normalised_action=action)  # offset is index 1 and 5, amplitude 0 and 4
#     sim.viewer(normalised_action=action)
    # sim.viewer(normalised_action=other_sol.parameters)
    # sim.plot_actions(normalised_action=other_sol.parameters)
    # sim.visualize_inner(generation=best_gen, episode=best_episode)
    # sim.finish(store=True, name="long_run_check_convergence")

    # best_solution, best_fitness = cma.search()

    # show_video(frame_generator=run_episode())