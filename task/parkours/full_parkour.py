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
        [-6.0, 2.0, 0.0],  # P2
        [-6.0, 0.0, 0.]   # P3
    ]))
    segment2 = BezierSegment(np.array([
        [-6.0, 0.0, 0.],  # P0
        [-6.0, -2.0, 0],  # P1
        [-6.0, -3.0, 0.],  # P2
        [-6.0, -4.0, 0.]   # P3
    ]))
    segment3 = BezierSegment(np.array([
        [-6.0, -4.0, 0.],  # P0
        [-6.0, -5.0, 0.],  # P1
        [-4.0, -5.0, 0.],  # P2
        [-2.0, -5.0, 0.]   # P3
    ]))
    segment4 = BezierSegment(np.array([
        [-2.0, -5.0, 0.],  # P0
        [0.0, -5.0, 0.],  # P1
        [0.0, -5.0, 2.],  # P2
        [-2.0, -5.0, 2.]   # P3
    ]))
    segment5 = BezierSegment(np.array([
        [-2.0, -5.0, 2.],  # P0
        [-4.0, -5.0, 2.],  # P1
        [-4.0, -5.0, 0.],  # P2
        [-2.0, -5.0, 0.]   # P3
    ]))
    segment6 = BezierSegment(np.array([
        [-2.0, -5.0, 0.],  # P0
        [0.0, -5.0, 0.],  # P1
        [5.0, -6.0, 0.],  # P2
        [5.0, 0.0, 0.]   # P3
    ]))



    parkour.add_qubic_bezier(new_segment=segment0)
    parkour.add_qubic_bezier(new_segment=segment1)
    parkour.add_qubic_bezier(new_segment=segment2)
    parkour.add_qubic_bezier(new_segment=segment3)
    parkour.add_qubic_bezier(new_segment=segment4)
    parkour.add_qubic_bezier(new_segment=segment5)
    parkour.add_qubic_bezier(new_segment=segment6)
    parkour.plot()
    parkour.store("task/parkours/full_parkour.pkl")