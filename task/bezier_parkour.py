import numpy as np
import plotly.graph_objects as go
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from typing import List

class BezierParkour:
    def __init__(self) -> None:
        self._segments: List[np.ndarray] = []

    def add_qubic_bezier(self, 
                    control_points: np.ndarray,
                    ) -> None:
        assert control_points.shape[0] == 4, "Control points must be 4 for a qubic Bézier curve"
        assert control_points.shape[1] == 3, "Control points must be 3D"

        if len(self._segments) == 0:
            self._segments.append(control_points)
        else:   # check for continuity 
            continuity: bool = np.allclose(self._segments[-1][-1, :] - self._segments[-1][-2, :], 
                                     control_points[1, :] - control_points[0, :]) and \
                            np.allclose(self._segments[-1][-1, :], control_points[0, :])
            if not continuity:
                prev_seg = self._segments[-1][-1, :] - self._segments[-1][-2, :]
                cur_seg = control_points[-1, :] - control_points[-2, :]
                s = "not continuous vector previous segement vector: "+str(prev_seg)+"\n vector current segment: "+str(cur_seg)+"\n or last and first point are not the same"
                warnings.warn(s)
                # TODO: adjust control points
            self._segments.append(control_points)
    
    def bezier_curve_segment(self, control_points, num_points=1000):
        """Compute a Bézier curve from control points."""
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 3))
        for i in range(n + 1):
            binom = comb(n, i)  # Compute the binomial coefficient
            curve += np.outer(binom * (t ** i) * ((1 - t) ** (n - i)), control_points[i])
        return curve
    
    def bezier_curve(self, num_points=1000):
        points = []
        for control_points in self._segments:
            points.append(self.bezier_curve_segment(control_points, num_points))
        return np.concatenate(points, axis=0)

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

        ax.legend()
        plt.show()



if __name__ == "__main__":
    parkour = BezierParkour()
    control_points1 = np.array([
        [0.0, 0.0, 0.],  # P0
        [0.0, 1.0, 0.],  # P1
        [1.0, 1.0, 0.5],  # P2
        [1.0, 0.0, 0.]   # P3
    ])

    control_points2 = np.array([
        [1.0, 0.0, 0.],  # P0
        [1.0, -1.0, -0.5],  # P1
        [0.0, -1.0, 0.],  # P2
        [0.0, 0.0, 0.]   # P3
    ])

    parkour.add_qubic_bezier(control_points=control_points1)
    parkour.add_qubic_bezier(control_points=control_points2)
    parkour.plot()


