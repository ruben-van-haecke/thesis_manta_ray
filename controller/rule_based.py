import numpy as np
from thesis_manta_ray.task.bezier_parkour import BezierParkour
from thesis_manta_ray.controller.quality_diversity import Archive
from scipy.spatial.transform import Rotation
from typing import List

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

def rotate(point: np.ndarray,
           rotation: np.ndarray) -> np.ndarray:
    """
    :param point: np.ndarray of shape (3,) for x, y, z
    :param rotation: np.ndarray of shape (3,) for roll, pitch, yaw
    :return: np.ndarray of shape (3,) for x, y, z
    """
    rot = Rotation.from_euler('xyz', rotation, degrees=False)
    point = rot.apply(point)
    return point

def translate_rotate(point: np.ndarray, 
                     translatation: np.ndarray,
                     rotation: np.ndarray) -> np.ndarray:
    """
    :param point: np.ndarray of shape (3,) for x, y, z
    :param translatation: np.ndarray of shape (3,) for x, y, z
        ie. to translate point w to the origin just give w as translation
    :param rotation: np.ndarray of shape (3,) for roll, pitch, yaw
    :return: np.ndarray of shape (3,) for x, y, z
    """
    point = point - translatation
    rot = Rotation.from_euler('xyz', rotation, degrees=False)
    point = rot.apply(point)
    return point

def chain_rot(point: np.ndarray,
              rotations: List[Rotation]
              ) -> np.ndarray:
    for rot in rotations:
        point = rot.apply(vectors=point)
    return point


class RuleBased:
    def __init__(self, archive: Archive):
        self._archive: Archive = archive
        self._rolling_features = np.zeros(3)

    def select_parameters_target(self, 
                          current_angular_positions: np.ndarray, 
                          current_xyz_velocities: np.ndarray,
                          current_position: np.ndarray,
                          target_location: np.ndarray,
                          print_flag: bool = False,
                          scaling: bool = True,
                          ) -> np.ndarray:
        """
        :param current_angular_positions: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_xyz_velocities: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_position: np.ndarray of shape (3,) for x, y, z
        :param target_location: np.ndarray of shape (num_points, 3) for x, y, z,
        
        :return: a tuple of:
            np.ndarray of shape (8,) which are the modulation parameters for the CPG, extracted from the archive
            np.ndarray of shape (3,) for roll, pitch, yaw, corresponding to the behaviour descriptor
        """
        #check if the transformation is correct
        current_angular_positions[0], current_angular_positions[1], current_angular_positions[2] = \
            -current_angular_positions[2], -current_angular_positions[1], -current_angular_positions[0]
        rot = Rotation.from_quat(current_angular_positions)
        target_location = target_location - current_position    # translate the target location such that the current position is the origin
        target_location_after_transformation = rot.apply(target_location, inverse=True)
        # rotation_after = rotate(point=[1, 0, 0], rotation=current_angular_positions)
        # rotation_after = rotate(point=rotation_after, rotation=-current_angular_positions)
        # print(f"rotation_after: {rotation_after}")
        # target_location_after_transformation = translate_rotate(point=target_location,
        #                  translatation=current_position,
        #                  rotation=current_angular_positions)
        # fish_location_after_transformation = translate_rotate(point=current_position,
        #                  translatation=current_position,
        #                  rotation=-current_angular_positions)
        # find the behaviour descriptor
        # roll
        roll = 0
        # pitch
        pitch = np.abs(np.arctan(target_location_after_transformation[2]/target_location_after_transformation[0]))
        pitch = np.pi - pitch if target_location_after_transformation[0] < 0 else pitch
        pitch = 0.4 * pitch/np.linalg.norm(target_location_after_transformation)
        if target_location_after_transformation[2] > 0: # negative pitch (there has been a rotation of 180 degrees around the z-axis)
            pitch *= -1

        # yaw
        yaw = np.abs(np.arctan(target_location_after_transformation[1]/target_location_after_transformation[0]))
        yaw = np.pi - yaw if target_location_after_transformation[0] < 0 else yaw
        yaw = 0.4 * yaw / np.linalg.norm(target_location_after_transformation)
        if target_location_after_transformation[1] < 0: # negative yaw
            yaw *= -1

        # scaling
        if scaling:
            m = max(np.abs(target_location_after_transformation[2]), 
                    np.linalg.norm(np.array([target_location_after_transformation[0], target_location_after_transformation[1]])))
            pitch = np.abs(target_location_after_transformation[2])/m * pitch
            yaw = np.linalg.norm(np.array([target_location_after_transformation[0], target_location_after_transformation[1]]))/m * yaw

            features = np.array([roll, pitch, yaw]).reshape(1, -1)
            self._rolling_features = self._rolling_features * 0.5 + features * 0.5
        else:
            roll, pitch, yaw = -roll, -pitch, yaw   # needed because of the difference in axes between te qd archive and the manta ray
            self._rolling_features = np.array([roll, pitch, yaw]).reshape(1, -1)

        sol = self._archive.get_closest_solutions(feature=self._rolling_features, k=1)[0][0]
        if print_flag:
            print(f"------------------------------------")
            print(f"orientation: {quat2euler(current_angular_positions)}")
            print(f"target_location_after_transformation: {target_location_after_transformation}")
            print(f"behaviour needed: {self._rolling_features}")
            print(f"behaviour selected: {sol.behaviour}, metadata: {sol.metadata}, bin_index: {self._archive.get_bin_index(features=sol.behaviour)}")

        # get parameters
        return sol.parameters, self._rolling_features

    def select_parameters_parkour(self, 
                          current_angular_positions: np.ndarray, 
                          current_xyz_velocities: np.ndarray,
                          current_position: np.ndarray,
                          parkour: BezierParkour,
                          print_flag: bool = False,
                          scaling: bool = True,
                          ) -> np.ndarray:
        """
        :param current_angular_positions: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_xyz_velocities: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_position: np.ndarray of shape (3,) for x, y, z
        :param future_points: np.ndarray of shape (num_points, 3) for x, y, z,
        
        :return: a tuple of:
            np.ndarray of shape (8,) which are the modulation parameters for the CPG, extracted from the archive
            np.ndarray of shape (3,) for roll, pitch, yaw, corresponding to the behaviour descriptor
        """
        distance_to_point, distance_parkour = parkour.get_distance(position=current_position)
        target_location = parkour.get_point(distance_parkour+1)
        scaled_action, behaviour_descriptor = self.select_parameters_target(current_angular_positions=current_angular_positions,
                                                        current_xyz_velocities=current_xyz_velocities,
                                                        current_position=current_position,
                                                        target_location=target_location,
                                                        print_flag=print_flag,
                                                        scaling=scaling
                                                        )

        return scaled_action, behaviour_descriptor