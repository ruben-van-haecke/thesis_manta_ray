from typing import Union
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart
from dm_control.mujoco.math import euler2quat
import numpy as np

from morphology.specification.specification import MantaRayMorphologySpecification

class MJCMantaRayTorso(MJCMorphologyPart):
    def __init__(self, parent: MJCMorphology | MJCMorphologyPart, name: str, pos: np.array, euler: np.array, *args, **kwargs) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)
    
    @property
    def morphology_specification(self) -> MantaRayMorphologySpecification:
        return super().morphology_specification
    
    @property
    def tail_position(self) -> np.ndarray:
        return np.array([self._torso_specification.length.value + self._torso_specification.radius.value, 0, 0])
    
    @property
    def pectoral_left_fin_position(self,
                                   ) -> np.ndarray:
        theta = 0# angle=from y-axis to z-axis
        return np.array([-self._torso_specification.length.value,
                          self._torso_specification.radius.value*np.cos(theta), 
                          self._torso_specification.radius.value*np.sin(theta), # angle=from y-axis to z-axis],
                          ])
    
    @property
    def pectoral_right_fin_position(self,
                                    ) -> np.ndarray:
        theta = 0# angle=from y-axis to z-axis
        return np.array([-self._torso_specification.length.value, 
                         -self._torso_specification.radius.value*np.cos(theta), # angle=from y-axis to z-axis
                          self._torso_specification.radius.value*np.sin(theta), # angle=from y-axis to z-axis]
                          ])
    
    @property
    def length_of_fin(self,
                        ) -> float:
            return self._torso_specification.length.value*2
    
    def _build(self) -> None:
        self._torso_specification = self.morphology_specification.torso_specification
        self._pectoral_fin_specificaiton = self.morphology_specification.pectoral_fin_specification
        self._build_torso()
    
    def _build_torso(self) -> None:
        r = self.morphology_specification.torso_specification.radius.value
        l = self.morphology_specification.torso_specification.length.value
        self.mjcf_body.add("geom",
                name=f"{self.base_name}_body",
                type="capsule",
                pos=np.zeros(3),
                euler=[0, np.pi / 2, 0],
                size=[r, l], #radius, length (half-length of the capsule)
                rgba=[100, 0, 0, 0.5],
                friction=[0.1, 0.1, 0.1],
                density=1000,   # default
                # fluidshape="ellipsoid",
                )