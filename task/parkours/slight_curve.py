from thesis_manta_ray.task.bezier_parkour import BezierParkour, BezierSegment
import numpy as np

if __name__ == "__main__":
    parkour = BezierParkour()
    segment0 = BezierSegment(np.array([
        [0.0, 0.0, 0.],  # P0
        [-1.0, 0.0, 0.],  # P1
        [-2.0, 0.0, 2.],  # P2
        [-3.0, 0.0, 0.]   # P3
    ]))
    segment1 = BezierSegment(np.array([
        [-3.0, 0.0, 0.],  # P0
        [-4.0, 0.0, -2.],  # P1
        [-5.0, 1.0, 0.5],  # P2
        [-6.0, 0.0, 0.]   # P3
    ]))


    parkour.add_qubic_bezier(new_segment=segment0)
    parkour.add_qubic_bezier(new_segment=segment1)
    parkour.plot()
    parkour.store("task/parkours/slight_curve.pkl")