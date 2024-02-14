from typing import Callable, List, Dict
from dm_control import composer
from dm_control.composer import Entity
from mujoco_utils.environment import MJCEnvironmentConfig
from mujoco_utils.robot import MJCMorphology
from mujoco_utils.observables import ConfinedObservable
from dm_control.mujoco.math import euler2quat
from dm_control.mjcf import Element

from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.specification import MantaRayMorphologySpecification 
from arena.hilly_light_aquarium import OceanArena
from dm_control import mjcf
from dm_control.composer.observation import observable

import numpy as np

class DragRaceTask(composer.Task):
    reward_functions = ["x_distance", "energy_efficient", "energy_efficient_velocity"]

    def __init__(self,
                 config: MJCEnvironmentConfig,
                 morphology: MJCMantaRayMorphology) -> None:
        super().__init__()
        self.config = config
        self._arena = self._build_arena()
        self._morphology: MJCMantaRayMorphology = self._attach_morphology(morphology)
        # self._configure_camera()
        self._task_observables = self._configure_observables()

        torso_radius = self._morphology.morphology_specification.torso_specification.radius.value
        self._initial_position = np.array([0.0, 0.0, 10 * torso_radius])
        self._configure_contacts()
        self._sensor_actuatorfrc_names = [sensor.name for sensor in self._morphology.mjcf_model.sensor.actuatorfrc]
        self._sensor_actuatorfrc = [sensor for sensor in self._morphology.mjcf_model.sensor.actuatorfrc]

        self._accumulated_energy = 0
    
    @property
    def root_entity(self) -> OceanArena:
        return self._arena
    
    @property
    def task_observables(
            self
            ) -> Dict:
        return self._task_observables
    
    def _build_arena(self) -> OceanArena:
        arena = OceanArena()
        return arena
    
    @staticmethod
    def _get_entity_xyz_position(
            entity: Entity,
            physics: mjcf.Physics
            ) -> np.ndarray:
        position, _ = entity.get_pose(
                physics=physics
                )
        return np.array(
                position
                )[:3]
    
    def _get_distance_from_initial_position(
            self,
            physics: mjcf.Physics
            ) -> float:
        morphology_position = self._get_entity_xyz_position(entity=self._morphology, physics=physics)
        distance = np.linalg.norm(morphology_position-self._initial_position)
        return distance
    
    def _get_x_distance_from_initial_position(
            self,
            physics: mjcf.Physics
            ) -> float:
        morphology_position = self._get_entity_xyz_position(entity=self._morphology, physics=physics)
        distance = np.linalg.norm(morphology_position[0]-self._initial_position[0])
        return distance
    
    def _configure_camera(self) -> None:
        self._arena.mjcf_root.worldbody.add(
                'camera', name='top_camera', target="torso", mode='targetbody', pos=[-1, -1, 4], quat=euler2quat(0, 0, 1))
    
    def _configure_contacts(
            self
            ) -> None:
        defaults = [self.root_entity.mjcf_model.default.geom, self._morphology.mjcf_model.default.geom]

        # Disable all collisions between geoms by default
        for geom_default in defaults:
            geom_default.contype = 1
            geom_default.conaffinity = 0
        # except for these one
        for geom in self._arena.needs_collision:
            geom.conaffinity = 1
    
    def _attach_morphology(self, morphology: MantaRayMorphologySpecification) -> MJCMantaRayMorphology:
        self._arena.add_free_entity(morphology)
        return morphology
    
    def _get_sensor_actuatorfrc(self,
                                physics: mjcf.Physics,
                                ) -> float:
        return physics.data.sensordata
    
    def _get_abs_forces_sensors(self,
                                physics: mjcf.Physics,
                                ) -> float:
        return np.sum(np.abs(self._get_sensor_actuatorfrc(physics=physics)))
    
    def _get_accumulated_energy_sensors(self,
                            physics: mjcf.Physics,
                            ) -> float:
        self._accumulated_energy += self._get_abs_forces_sensors(physics=physics) * self.config.physics_timestep     # d energy = f * dt
        return self._accumulated_energy
    
    def _configure_task_observables(
            self
            ) -> Dict[str, observable.Observable]:
        task_observables = dict()
        task_observables["task/time"] = ConfinedObservable(
                low=0,
                high=self.config.simulation_time,
                shape=[1],
                raw_observation_callable=lambda
                    physics: physics.time()
                )
        task_observables["task/xyz-distance-from-origin"] = ConfinedObservable(
                low=0, high=np.inf, shape=[1], raw_observation_callable=self._get_distance_from_initial_position
                )
        task_observables["task/force"] = ConfinedObservable(
                low=0, high=np.inf, shape=[4], raw_observation_callable=self._get_sensor_actuatorfrc
                )
        # task_observables["task/xy-distance-to-target"] = ConfinedObservable(
        #         low=0, high=np.inf, shape=[1], raw_observation_callable=self._get_xy_distance_to_target
        #         )
        # task_observables["task/xy-direction-to-target"] = ConfinedObservable(
        #         low=-1, high=1, shape=[2], raw_observation_callable=self._get_xy_direction_to_target
        #         )

        for obs in task_observables.values():
            obs.enabled = True

        return task_observables
    
    def _configure_observables(self) -> Dict[str, observable.Observable]:
        # self._configure_morphology_observables()
        task_observables = self._configure_task_observables()
        return task_observables
    
    def get_reward(self, physics):
        "reward to minimize"
        v = self.config.velocity
        if self.config.reward_fn == "x_distance" or self.config.reward_fn is None:
            return self._get_x_distance_from_initial_position(physics=physics)
        elif self.config.reward_fn == "energy_efficient":
            return self._get_accumulated_energy_sensors(physics=physics)/current_distance_from_initial_position
        elif self.config.reward_fn == "energy_efficient_velocity":
            current_distance_from_initial_position = self._get_x_distance_from_initial_position(physics=physics)
            if current_distance_from_initial_position == 0.:
                return 1/0.00001
            velocity_penalty = np.abs(v*physics.time() - current_distance_from_initial_position)
            return (self._get_accumulated_energy_sensors(physics=physics)+200*velocity_penalty)/current_distance_from_initial_position
        else:
            raise ValueError("reward_fn not recognized")
    
    def _initialize_morphology_pose(
            self,
            physics: mjcf.Physics
            ) -> None:
        initial_quaternion = euler2quat(0, 0, 0)

        self._morphology.set_pose(
                physics=physics, position=self._initial_position, quaternion=initial_quaternion
                )

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._initialize_morphology_pose(physics)
    


class Move(MJCEnvironmentConfig):
    def __init__(
            self, 
            seed: int = 42, 
            time_scale: float = 1, 
            control_substeps: int = 1, 
            simulation_time: float = 10,
            camera_ids: List[int] | None = None,
            velocity: float = 0.5,
            reward_fn: str | None = None,
            ) -> None:
        super().__init__(
            task = DragRaceTask, 
            time_scale=time_scale,
            control_substeps=control_substeps,
            simulation_time=simulation_time,
            camera_ids=[0, 1],
        )
        self.velocity = velocity
        self.reward_fn = reward_fn