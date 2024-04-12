from copy import deepcopy
import math
from typing import List, Dict
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
from scipy.spatial.transform import Rotation

import numpy as np

from task.bezier_parkour import BezierParkour

def quat2euler(q):
    """
    Convert an array of quaternions into Euler angles (roll, pitch, yaw).

    :param quaternions: NumPy array of shape (4,) containing quaternions.
    :return: NumPy array of shape (3, ) containing Euler angles.
    x (roll) is in the range -pi to pi
    y (pitch) is in the range -pi/2 to pi/2
    z (yaw) is in the range -pi to pi
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


class MoveConfig(MJCEnvironmentConfig):
    def __init__(
            self, 
            seed: int = 42, 
            time_scale: float = 1, 
            control_substeps: int = 1, 
            simulation_time: float = 10,
            camera_ids: List[int] | None = None,
            velocity: float = 0.5,
            reward_fn: str | None = None,
            task_mode: str = "parkour",
            parkour: BezierParkour | None = None,
            ) -> None:
        super().__init__(
            task = Task, 
            time_scale=time_scale,
            control_substeps=control_substeps,
            simulation_time=simulation_time,
            camera_ids=[0, 1],
        )
        if task_mode != 'parkour' and (parkour is not None):
            raise ValueError("parkour may only be defined if task_mode is parkour")
        self._velocity = velocity
        self._reward_fn = reward_fn
        self._task_mode = task_mode
        self._parkour = parkour
        self._location_target = None
        self._initial_position = np.array([0.0, 0.0, 15 * 0.1])
    
    def __deepcopy__(self, memo) -> 'MoveConfig':
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    @property
    def initial_position(self) -> np.ndarray:
        return self._initial_position
    
    @property
    def velocity(self) -> float:
        return self._velocity
    
    @property
    def reward_fn(self) -> str | None:
        return self._reward_fn
    
    @property
    def task_mode(self) -> str:
        return self._task_mode
    
    @property
    def parkour(self) -> BezierParkour | None:
        return self._parkour
    
    @property
    def target_location(self):
        if self._task_mode == "random_target":
            return self._location_target
        else:
            raise ValueError("task_mode is not random_target")
    @target_location.setter
    def target_location(self, location: np.ndarray):
        if self._task_mode == "random_target":
            self._location_target = location
        else:
            raise ValueError("task_mode is not random_target")
    

class Task(composer.Task):
    reward_functions = ["Δx", "E * Δx", "(E + 200*Δx) * (Δx)"]

    def __init__(self,
                 config: MoveConfig,#MJCEnvironmentConfig,
                 morphology: MJCMantaRayMorphology, 
                 ) -> None:
        super().__init__()
        
        self._config = config
        self._arena: OceanArena = self._build_arena()
        self._morphology: MJCMantaRayMorphology = self._attach_morphology(morphology)
        if self._config.task_mode == "parkour":
            # self._parkour_entities = self._arena._build_parkour_line()
            pass
        elif self._config.task_mode == "random_target":
            pass
        # self._configure_camera()
        self._task_observables = self._configure_observables()

        torso_radius = self._morphology.morphology_specification.torso_specification.radius.value
        self._initial_position = self._config.initial_position  #np.array([0.0, 0.0, 10 * torso_radius])
        self._configure_contacts()
        self._sensor_actuatorfrc_names = [sensor.name for sensor in self._morphology.mjcf_model.sensor.actuatorfrc]
        self._sensor_actuatorfrc = [sensor for sensor in self._morphology.mjcf_model.sensor.actuatorfrc]

        self._accumulated_energy = 0
        self._accumulated_distance = 0
        time_window = 2 # number of seconds to calculate the delta orientation
        # self._previous_orientations = np.zeros((4, int(time_window/self._config.control_timestep)))
        self._previous_position = self._initial_position
        self._angular_velocity_sum = np.zeros(3)
        self._angular_velocity_num = 0
        self._orientation_iterator = 0
    
    @property
    def root_entity(self) -> OceanArena:
        return self._arena
    
    @property
    def task_observables(
            self
            ) -> Dict:
        return self._task_observables
    
    def _build_arena(self) -> OceanArena:
        arena = OceanArena(task_mode=self._config.task_mode, 
                           initial_position=self._config.initial_position,
                           parkour=self._config.parkour)
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
    
    def _get_distance_travelled(
            self,
            physics: mjcf.Physics
            ) -> float:
        morphology_position = self._get_entity_xyz_position(entity=self._morphology, physics=physics)
        self._accumulated_distance += np.linalg.norm(morphology_position-self._previous_position)
        self._previous_position = morphology_position
        return self._accumulated_distance

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
    
    def _get_orientation_entity(self,
                                entity: Entity,
                                physics: mjcf.Physics,
                                ) -> float:
        pose = entity.get_pose(physics=physics) # (position, quaternion) with quaternion [w, x, y, z]
        return quat2euler(pose[1])  # new_euler
    
    def _get_orientation(self,
                         physics: mjcf.Physics,
                         ) -> float:
        orientation = self._get_orientation_entity(entity=self._morphology, physics=physics)
        return orientation
    
    def _get_position(self,
                      physics: mjcf.Physics,
                      ) -> float:
        return self._get_entity_xyz_position(entity=self._morphology, physics=physics)
    
    def _get_xyz_velocity(self,
                          physics: mjcf.Physics,
                        ) -> float:
        lin_vel, angular_velocity = self._morphology.get_velocity(physics=physics)
        return lin_vel
    
    def _get_angular_velocity(self,
                              physics: mjcf.Physics,
                              ) -> float:
        angular_velocity = np.array(self._morphology.get_velocity(physics=physics)[1])
        return angular_velocity
    
    def _get_avg_angular_velocity(self, 
                              physics: mjcf.Physics,
                              ) -> float:
        # if physics.time() < 2:
        #     return np.nan
        lin_vel, angular_velocity = self._morphology.get_velocity(physics=physics) # (position, quaternion) with quaternion [w, x, y, z]
        angular_vel = np.array(angular_velocity)
        self._angular_velocity_sum += angular_vel
        self._angular_velocity_num += 1
        return self._angular_velocity_sum/self._angular_velocity_num
        
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
        task_observables["task/position"] = ConfinedObservable(
                low=-np.inf, high=np.inf, shape=[3], raw_observation_callable=self._get_position
                )
        task_observables["task/xyz_velocity"] = ConfinedObservable(
                low=-np.inf, high=np.inf, shape=[3], raw_observation_callable=self._get_xyz_velocity
                )
        task_observables["task/angular_velocity"] = ConfinedObservable(
                low=-np.inf, high=np.inf, shape=[3], raw_observation_callable=self._get_angular_velocity
                )
        task_observables["task/avg_angular_velocity"] = ConfinedObservable(
                low=-np.inf, high=np.inf, shape=[3], raw_observation_callable=self._get_avg_angular_velocity
                )
        task_observables["task/accumulated_energy"] = ConfinedObservable(
                low=0, high=np.inf, shape=[1], raw_observation_callable=self._get_accumulated_energy_sensors
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
        assert self._config.reward_fn in Task.reward_functions, f"reward_fn not recognized, choose from {Task.reward_functions}"

        v = self._config.velocity
        current_distance_from_initial_position = self._get_distance_travelled(physics=physics)
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
        self._accumulated_distance = 0
        self._angular_velocity_sum = 0
        self._angular_velocity_num = 0
        if self._config.task_mode == "random_target":
            self._arena.set_target_location(physics=physics, position=self._config.target_location)
        elif self._config.task_mode == "parkour":
            pass
        
    
if __name__ == '__main__':
    # roll = np.linspace(-np.pi, np.pi, 100)
    # pitch = np.linspace(-np.pi, np.pi, 100)
    # yaw = np.linspace(-np.pi, np.pi, 100)
    # for r in roll:
    #     for p in pitch:
    #         for y in yaw:
    #             q = euler2quat(np.degrees(r), np.degrees(p), np.degrees(y))
    #             e = quat2euler(q)
    #             rpy = np.array([r, p, y])
    #             for i in range(3):
    #                 assert np.allclose(rpy[i], e[i]) or \
    #                     np.allclose(min(rpy[i], e[i]), max(rpy[i], e[i])-np.pi), f"original: {rpy}, q: {q}, e: {e}"
    from scipy.spatial.transform import Rotation
    angle = Rotation.from_quat([0, 0, 0, 1]).as_euler('xyz', degrees=False)
    print(f"angle: {angle}")
    quaternion = Rotation.from_euler('xyz', angle, degrees=True).as_quat()
    print(f"quaternion: {quaternion}")
    angle = Rotation.from_quat(quaternion).as_euler('xyz', degrees=False)
    print(f"angle: {angle}")
