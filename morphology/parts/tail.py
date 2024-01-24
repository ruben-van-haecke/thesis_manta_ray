import numpy as np
from typing import Union

from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart
from specification.specification import MantaRayMorphologySpecification
from parts.tail_segment import MJCMantaRayTailSegment

class MJCMantaRayTail(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array, *args, **kwargs) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)
    
    @property
    def morphology_specification(self) -> MantaRayMorphologySpecification:
        return super().morphology_specification
    
    def _build(self, *args, **kwargs) -> None:
        self._tail_specification = self.morphology_specification.tail_specification
        self._torso_specification = self.morphology_specification.torso_specification

        self._build_segments()
    
    def _build_segments(self) -> None:
        self._segments = []
        number_of_segments = len(self._tail_specification.segment_specifications)
        for segment_index in range(number_of_segments):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
            except IndexError:  # first segment
                position = [0, 0, 0]#np.zeros(3)
                parent = self

            segment = MJCMantaRayTailSegment(
                    parent=parent,
                    name=f"{self.base_name}_segment_{segment_index}",
                    pos=position,
                    euler=np.zeros(3),
                    segment_index=segment_index
                    )
            self._segments.append(segment)