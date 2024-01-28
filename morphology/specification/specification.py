import numpy as np
from typing import List
from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class MantaRayPectoralFinJointSpecification(Specification):
    def __init__(self,
                 range: float,
                 stiffness: float,
                 damping: float,
                 ) -> None:
        super().__init__()
        self.range = FixedParameter(value=range)
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)

class MantaRayPectoralFinTendonSpecification(Specification):
    def __init__(self,
                 range_deviation: float,
                 stiffness: float,
                 damping: float,
                 ) -> None:
        super().__init__()
        self.range_deviation = FixedParameter(value=range_deviation)
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)

class MantaRayTailJointSpecification(Specification):
    def __init__(self,
                 range: float,
                 stiffness: float,
                 damping: float,
                 ) -> None:
        super().__init__()
        self.range = FixedParameter(value=range)
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)

class MantaRayActuationSpecification(Specification):
    def __init__(self,
                 kp: float,
                 ) -> None:
        super().__init__()
        self.kp = FixedParameter(value=kp)


class MantaRayTorsoSpecification(Specification):
    def __init__(self,
                 r_: float,
                 l_: float) -> None:
        super().__init__()
        self.radius = FixedParameter(r_) # radius of the torso
        self.length = FixedParameter(l_) # half-length of the torso


class MantaRayTailSegmentSpecification(Specification):
    def __init__(self,
                 radius: float,
                 length: float,
                 joint_specifications: MantaRayTailJointSpecification,
                 ) -> None:
        super().__init__()
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.joint_specification = joint_specifications


class MantaRayTailSpecification(Specification):
    def __init__(self,
                 segment_specifications: List[MantaRayTailSegmentSpecification],
                 ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications

class MantaRayPectoralFinSegmentSpecification(Specification):
    def __init__(self,
                 radius: float,
                 max_length: float,
                 joint_specification: MantaRayPectoralFinJointSpecification,
                 segment_index: int,
                 amount_of_segments: int,
                 ) -> None:
        super().__init__()
        length = max_length - max_length*((segment_index)/amount_of_segments)**(0.5)
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.joint_specification = joint_specification

class MantaRayPectoralFinSpecification(Specification):
    def __init__(self,
                 segment_specifications: List[MantaRayPectoralFinSegmentSpecification],
                 joint_specification: MantaRayPectoralFinJointSpecification,
                 tendon_specification: MantaRayPectoralFinTendonSpecification,
                 ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications
        self.number_of_segments = len(segment_specifications)
        self.joint_specification = joint_specification
        self.tendon_specification = tendon_specification



class MantaRayMorphologySpecification(MorphologySpecification):
    def __init__(self,
                 torso_specification: MantaRayTorsoSpecification,
                 tail_specification: MantaRayTailSpecification,
                 pectoral_fin_specification: MantaRayPectoralFinSpecification,
                 actuation_specification: MantaRayActuationSpecification,
                 ) -> None:
        super().__init__()
        self.torso_specification = torso_specification
        self.tail_specification = tail_specification
        self.pectoral_fin_specification = pectoral_fin_specification
        self.actuation_specification = actuation_specification    # taken from toy example
    