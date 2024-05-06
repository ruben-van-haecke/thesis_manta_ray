import numpy as np
from thesis_manta_ray.task.bezier_parkour import BezierParkour
from thesis_manta_ray.controller.quality_diversity import Archive
from scipy.spatial.transform import Rotation


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
        target_location_after_transformation = translate_rotate(point=target_location,
                         translatation=current_position,
                         rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        fish_location_after_transformation = translate_rotate(point=current_position,
                         translatation=current_position,
                         rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        # find the behaviour descriptor
        # roll
        roll = 0
        # pitch
        pitch = np.arctan(np.abs(target_location_after_transformation[2]/target_location_after_transformation[0]))
        pitch += np.pi/2 if target_location_after_transformation[0] < 0 else 0
        if target_location_after_transformation[2] < 0: # negative pitch (there has been a rotation of 180 degrees around the z-axis)
            pitch *= -1
        # yaw
        yaw = np.arctan(np.abs(target_location_after_transformation[1]/target_location_after_transformation[0]))
        yaw += np.pi/2 if target_location_after_transformation[0] < 0 else 0
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
            self._rolling_features = np.array([roll, pitch, yaw]).reshape(1, -1)


        if print_flag:
            print(f"------------------------------------")
            print(f"angular_position: {current_angular_positions}")
            print(f"target_location_after_transformation: {target_location_after_transformation}")
            print(f"rolling_features: {self._rolling_features}")

        # get parameters
        sol = self._archive.get_closest_solutions(feature=self._rolling_features, k=1)[0][0]
        return sol.parameters, self._rolling_features

    def select_parameters_parkour(self, 
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
        
        :return: a tuple of:
            np.ndarray of shape (8,) which are the modulation parameters for the CPG, extracted from the archive
            np.ndarray of shape (3,) for roll, pitch, yaw, corresponding to the behaviour descriptor
        """
        # get the current roll, pitch, yaw
        roll, pitch, yaw = current_angular_positions
        # define the angular velocity in radians per second for the roll, pitch, yaw based on current_angular_velocities and future_points
        distance, parkour_distance = parkour.get_distance(current_position)
        # get point
        point_parkour = parkour.get_point(parkour_distance+0.5)
        point_parkour_transformed = translate_rotate(point=point_parkour,
                                 translatation=current_position,
                                 rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        print(f"point_parkour_transformed: {point_parkour_transformed}")
        current_position_transformed = translate_rotate(point=current_position,
                                                        translatation=current_position,
                                                        rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        v_perpendicular = point_parkour_transformed - current_position_transformed
        v_perpendicular /= np.linalg.norm(v_perpendicular)  # normalize
        v_path = parkour.get_tangent(parkour_distance)
        v_path = rotate(v_path, current_angular_positions)
        v_next = v_path + distance*v_perpendicular
        # get rotation
        current_xyz_velocities_transformed = rotate(current_xyz_velocities, -current_angular_positions)
        current_xz_velocities = current_xyz_velocities_transformed[[0, 2]] / np.linalg.norm(current_xyz_velocities_transformed[[0, 2]])
        z_angle_current = np.arcsin(current_xz_velocities[1])
        v_next_xz = v_next[[0, 2]] / np.linalg.norm(v_next[[0, 2]])
        z_angle_next = np.arcsin(v_next_xz[1])
        z_angle_diff = z_angle_next - z_angle_current   # pitch difference
        pitch = z_angle_diff

        current_xy_velocities = current_xyz_velocities_transformed[[0, 1]] / np.linalg.norm(current_xyz_velocities_transformed[[0, 1]])
        y_angle_current = np.arcsin(current_xy_velocities[1])
        v_next_xy = v_next[[0, 1]] / np.linalg.norm(v_next[[0, 1]])
        y_angle_next = np.arcsin(v_next_xy[1])
        y_angle_diff = y_angle_next - y_angle_current   # yaw difference
        yaw = y_angle_diff

        roll = 0

        # rot, rssd = Rotation.align_vectors(a=v_next.reshape((1, 3)), b=current_xyz_velocities.reshape((1, 3))) # vector b to vector a
        # roll, pitch, yaw = rot.as_euler('xyz')
        # roll, pitch, yaw = parkour.get_rotation(parkour_distance)
        # yaw -= np.pi
        features = np.array([roll, pitch, yaw]).reshape(1, -1)
        # get the parameters from the archive
        # parameters = self._archive.interpolate(features=features)
        sol = self._archive.get_closest_solutions(feature=features, k=1)[0][0]
        print(f"features: {features}")
        print(f"current_angular_positions: {current_angular_positions}")
        parameters = sol.parameters
        if np.any(parameters>1) or np.any(parameters<0):
            print("Parameters out of bounds")
            print(f"Parameters: {parameters}")
            parameters = np.clip(parameters, 0, 1)

        return parameters, features