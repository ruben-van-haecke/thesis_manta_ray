import numpy as np
import plotly.graph_objects as go
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.spatial.transform import Rotation
from typing import List
import pickle

class BezierSegment:
    def __init__(self, control_points: np.ndarray) -> None:
        assert control_points.shape[0] == 4, "Control points must be 4 for a qubic Bézier curve"
        assert control_points.shape[1] == 3, "Control points must be 3D"
        self._control_points = control_points
        self._lut = self.create_lookup_table()

    def __getitem__(self, index):
        return self.control_points[index]
    
    @property
    def lut(self):
        return self._lut

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        assert value.shape[0] == 4, "Control points must be 4 for a cubic Bézier curve"
        assert value.shape[1] == 3, "Control points must be 3D"
        self._control_points = value

    def bezier_curve(self, num_points=1000):
        n = len(self._control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 3))
        for i in range(n + 1):
            binom = comb(n, i)  # Compute the binomial coefficient
            curve += np.outer(binom * (t ** i) * ((1 - t) ** (n - i)), self._control_points[i])
        return curve
    
    def create_lookup_table(self, num_points=1000):
        curve = self.bezier_curve(num_points)
        lengths = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        cumulative_lengths = np.cumsum(lengths)
        lookup_table = dict(zip(cumulative_lengths, curve))
        return lookup_table
    
    def get_point(self, distance: float) -> np.ndarray:
        # Find the closest distance in the lookup table
        closest_distance = min(self._lut.keys(), key=lambda x: abs(x - distance))
        # Return the corresponding point
        return self._lut[closest_distance]
    
    def rotate(self, rot: Rotation) -> None:
        self._control_points = rot.apply(self._control_points)
        # remake the lut
        self._lut = self.create_lookup_table()
    
    def translate(self, translation: np.ndarray) -> None:
        self._control_points += translation
        # remake the lut
        self._lut = self.create_lookup_table()
    
    


class BezierParkour:
    def __init__(self) -> None:
        self._segments: List[BezierSegment] = []
        self._lut = self._create_lookup_table()
    
    def get_distance(self, position: np.ndarray) -> float:
        """
        Get the distance to the closest point on the parkour, given a position. 
        Followed by the distance from the start of the parkour.
        """
        distance, point = min(self._lut.items(), key=lambda dist, point: np.linalg.norm(point - position))
        return np.linalg.norm(position - point), distance
    
    def get_rotation(self, distance: float) -> np.ndarray:
        """Get the rotation: roll, pitch, yawn at a certain distance"""
        # Find the closest point in the lookup table and the second closest point
        closest_distance = min(self._lut.keys(), key=lambda x: abs(x - distance))
        second_closest_distance = min(self._lut.keys(), key=lambda dist: dist - closest_distance if dist - closest_distance > 0 else np.inf)
        # Compute the direction vector between the two points
        direction = self._lut[second_closest_distance] - self._lut[closest_distance]
        return Rotation.align_vectors(direction, np.array([-1, 0, 0])).as_euler('xyz')   # transforms the negative x-axis to the direction vector
    
    def _create_lookup_table(self, num_points=1000):
        lookup_table = dict()
        max_distance = 0
        for segment in self._segments:
            for key, value in segment.lut.items():
                lookup_table[max_distance + key] = value
            max_distance += max(segment.lut.keys())
        return lookup_table

    def add_qubic_bezier(self, 
                    new_segment: BezierSegment,
                    ) -> None:
        if len(self._segments) == 0:
            self._segments.append(new_segment)
        else:   # check for continuity 
            continuity: bool = np.allclose(self._segments[-1][-1, :] - self._segments[-1][-2, :], 
                                     new_segment[1, :] - new_segment[0, :]) and \
                            np.allclose(self._segments[-1][-1, :], new_segment[0, :])
            if not continuity:
                prev_seg = self._segments[-1][-1, :] - self._segments[-1][-2, :]
                cur_seg = new_segment[-1, :] - new_segment[-2, :]
                s = "not continuous vector previous segement vector: "+str(prev_seg)+"\n vector current segment: "+str(cur_seg)+"\n or last and first point are not the same"
                warnings.warn(s)
                # TODO: adjust control points
            self._segments.append(new_segment)
        self._lut = self._create_lookup_table()
    
    def bezier_curve(self, num_points=1000):
        """
        Compute the Bézier curve for the entire parkour
        returns: np.ndarray of shape (num_points * num_segments, 3)
        """
        points = []
        distances = np.linspace(0, max(self._lut.keys()), num_points)
        for distance in distances:
            point = self.get_point(distance)
            points.append(point)
        points = np.array(points)
        return points
    
    def get_point(self, distance: float) -> np.ndarray:
        # Find the closest distance in the lookup table
        closest_distance = min(self._lut.keys(), key=lambda x: abs(x - distance))
        # Return the corresponding point
        return self._lut[closest_distance]
    
    def allign_points(self, 
                      vector: np.ndarray,
                      points: np.ndarray) -> np.ndarray:
        """Allign points to the same plane"""
        assert vector.shape[0] == 3, "Vector must be 3D"
        assert points.shape[1] == 3, "Points must be 3D"
        assert points.shape[0] == 4, "At least 4 points are needed for the qubiq bezier curve"
        
        raise NotImplementedError
    

    def plot(self) -> None:
        # Compute the curve points
        curve = self.bezier_curve()
        # Plotting the curve
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label="Bézier Curve")

        # Plot the control points
        for control_points in self._segments:
            ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro-', label="Control Points")

        # Set the axes' aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])
        # Set labels for x, y, and z axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        plt.show()

    def store(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(path: str) -> 'BezierParkour':
        with open(path, 'rb') as file:
            return pickle.load(file)



if __name__ == "__main__":
    parkour = BezierParkour()
    segment0 = BezierSegment(np.array([
        [0.0, 0.0, 0.],  # P0
        [1.0, 0.0, 0.],  # P1
        [2.0, 0.0, 0.],  # P2
        [3.0, 0.0, 0.]   # P3
    ]))
    segment1 = BezierSegment(np.array([
        [0.0, 0.0, 0.],  # P0
        [0.0, 1.0, 0.],  # P1
        [1.0, 1.0, 0.5],  # P2
        [1.0, 0.0, 0.]   # P3
    ]))

    segment2 = BezierSegment(np.array([
        [1.0, 0.0, 0.],  # P0
        [1.0, -1.0, -0.5],  # P1
        [0.0, -1.0, 0.],  # P2
        [0.0, 0.0, 0.]   # P3
    ]))

    parkour.add_qubic_bezier(new_segment=segment1)
    parkour.add_qubic_bezier(new_segment=segment2)
    parkour.plot()


