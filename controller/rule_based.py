import numpy as np
from thesis_manta_ray.task.bezier_parkour import BezierParkour
from thesis_manta_ray.controller.quality_diversity import Archive
from scipy.spatial.transform import Rotation

class RuleBased:
    def __init__(self, archive: Archive):
        self._archive: Archive = archive

    def select_parameters(self, 
                          current_angular_positions: np.ndarray, 
                          current_xyz_velocities: np.ndarray,
                          current_position: np.ndarray,
                          parkour: BezierParkour,
                          ) -> np.ndarray:
        """
        :param current_angular_positions: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_xyz_velocities: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_position: np.ndarray of shape (3,) for x, y, z
        :param future_points: np.ndarray of shape (num_points, 3) for x, y, z,
        
        :return: np.ndarray of shape (8,) which are the modulation parameters for the CPG, extracted from the archive"""
        # get the current roll, pitch, yaw
        roll, pitch, yaw = current_angular_positions
        # get the current position
        x, y, z = current_position
        # define the angular velocity in radians per second for the roll, pitch, yaw based on current_angular_velocities and future_points
        distance, parkour_distance = parkour.get_distance(current_position)
        # get point
        point = parkour.get_point(parkour_distance)
        v_perpendicular = point - current_position
        v_perpendicular /= np.linalg.norm(v_perpendicular)  # normalize
        v_path = parkour.get_tangent(parkour_distance)
        v_next = v_path + distance*v_perpendicular
        # get rotation
        
        rot, rssd = Rotation.align_vectors(a=v_next.reshape((1, 3)), b=current_xyz_velocities.reshape((1, 3))) # vector b to vector a
        roll, pitch, yaw = rot.as_euler('xyz')
        # roll, pitch, yaw = parkour.get_rotation(parkour_distance)
        # yaw -= np.pi
        features = np.array([roll, pitch, yaw]).reshape(1, -1)
        # get the parameters from the archive
        # parameters = self._archive.interpolate(features=features)
        parameters = self._archive.get_closest_solutions(feature=features, k=1)[0][0].parameters
        if np.any(parameters>1) or np.any(parameters<0):
            print("Parameters out of bounds")
            print(f"Parameters: {parameters}")
            parameters = np.clip(parameters, 0, 1)

        return parameters