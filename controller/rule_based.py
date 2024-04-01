import numpy as np
from thesis_manta_ray.task.bezier_parkour import BezierParkour
from thesis_manta_ray.controller.quality_diversity import Archive

class RuleBased:
    def __init__(self, archive: Archive):
        self._archive: Archive = archive

    def select_parameters(self, 
                          current_angular_positions: np.ndarray, 
                          current_position: np.ndarray,
                          parkour: BezierParkour,
                          ) -> np.ndarray:
        """
        :param current_angular_velocities: np.ndarray of shape (3,) for roll, pitch, yaw
        :param current_position: np.ndarray of shape (3,) for x, y, z
        :param future_points: np.ndarray of shape (num_points, 3) for x, y, z,
        
        :return: np.ndarray of shape (8,) which are the modulation parameters for the CPG, extracted from the archive"""
        # get the current roll, pitch, yaw
        roll, pitch, yaw = current_angular_positions
        # get the current position
        x, y, z = current_position
        # define the angular velocity in radians per second for the roll, pitch, yaw based on current_angular_velocities and future_points
        distance, traversed_distance = parkour.get_distance(current_position)
        # get point
        point = parkour.get_point(traversed_distance)
        # get rotation
        roll, pitch, yaw = parkour.get_rotation(traversed_distance)
        # get the parameters from the archive
        parameters = self._archive.interpolate(features=np.array([roll, pitch, yaw]))

        return parameters