from typing import Callable, List, Dict
from dm_control import composer
from dm_control.composer import Entity
from mujoco_utils.environment import MJCEnvironmentConfig
from mujoco_utils.robot import MJCMorphology
from mujoco_utils.observables import ConfinedObservable
from dm_control.mujoco.math import euler2quat
from dm_control.mjcf import Element
from arena.entities.target import Target

from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.specification import MantaRayMorphologySpecification 
from arena.hilly_light_aquarium import OceanArena
from dm_control import mjcf
from dm_control.composer.observation import observable

import numpy as np

def quat2euler(q):
    """
    Convert an array of quaternions into Euler angles (roll, pitch, yaw).

    :param quaternions: NumPy array of shape (4,) containing quaternions.
    :return: NumPy array of shape (3, ) containing Euler angles.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))  # Clip sinp for numerical stability

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


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
            task_mode: str = "parkour"
            ) -> None:
        super().__init__(
            task = DragRaceTask, 
            time_scale=time_scale,
            control_substeps=control_substeps,
            simulation_time=simulation_time,
            camera_ids=[0, 1],
        )
        self._velocity = velocity
        self._reward_fn = reward_fn
        self._task_mode = task_mode
    @property
    def velocity(self) -> float:
        return self._velocity
    
    @property
    def reward_fn(self) -> str | None:
        return self._reward_fn
    
    @property
    def task_mode(self) -> str:
        return self._task_mode
    

class DragRaceTask(composer.Task):
    reward_functions = ["Δx", "E * Δx", "(E + 200*Δx) * (Δx)"]

    def __init__(self,
                 config: Move,#MJCEnvironmentConfig,
                 morphology: MJCMantaRayMorphology) -> None:
        super().__init__()
        
        self._config = config
        self._arena = self._build_arena()
        self._morphology: MJCMantaRayMorphology = self._attach_morphology(morphology)
        if self._config.task_mode == "parkour":
            self._parkour = self._attach_parkour()
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
        arena = OceanArena(task_mode=self._config.task_mode)
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
    
    def _attach_parkour(self) -> List[Element]:
        parkour = []
        t = Target()
        parkour.append(t)
        self._arena.attach(t)
        return parkour
    
    def _get_sensor_actuatorfrc(self,
                                physics: mjcf.Physics,
                                ) -> float:
        return physics.data.sensordata
    
    def _get_orientation_entity(self,
                                entity: Entity,
                                physics: mjcf.Physics,
                                ) -> float:
        pose = entity.get_pose(physics=physics)
        return quat2euler(pose[1])
    
    def _get_orientation(self,
                         physics: mjcf.Physics,
                         ) -> float:
        orientation = self._get_orientation_entity(entity=self._morphology, physics=physics)
        return orientation
        
    def _get_abs_forces_sensors(self,
                                physics: mjcf.Physics,
                                ) -> float:
        return np.sum(np.abs(self._get_sensor_actuatorfrc(physics=physics)))
    
    def _get_accumulated_energy_sensors(self,
                            physics: mjcf.Physics,
                            ) -> float:
        self._accumulated_energy += self._get_abs_forces_sensors(physics=physics) * self._config.physics_timestep     # d energy = f * dt
        return self._accumulated_energy
    
    def _configure_task_observables(
            self
            ) -> Dict[str, observable.Observable]:
        task_observables = dict()
        task_observables["task/time"] = ConfinedObservable(
                low=0,
                high=self._config.simulation_time,
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
        task_observables["task/orientation"] = ConfinedObservable(
                low=-np.pi, high=np.pi, shape=[3], raw_observation_callable=self._get_orientation
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
        assert self._config.reward_fn in DragRaceTask.reward_functions, f"reward_fn not recognized, choose from {DragRaceTask.reward_functions}"

        v = self._config.velocity
        current_distance_from_initial_position = self._get_distance_from_initial_position(physics=physics)
        velocity_penalty = np.abs(v*physics.time() - current_distance_from_initial_position)

        if self._config.reward_fn == "Δx" or self._config.reward_fn is None:
            return velocity_penalty
        elif self._config.reward_fn == "E * Δx":
            return self._get_accumulated_energy_sensors(physics=physics)*velocity_penalty #/current_distance_from_initial_position
        elif self._config.reward_fn == "(E + 200*Δx) * (Δx)":
            if current_distance_from_initial_position == 0.:
                return 1/0.00001
            return (self._get_accumulated_energy_sensors(physics=physics)+200*velocity_penalty)*velocity_penalty #+200*velocity_penalty)/current_distance_from_initial_position
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
        self._accumulated_energy = 0
        
    
