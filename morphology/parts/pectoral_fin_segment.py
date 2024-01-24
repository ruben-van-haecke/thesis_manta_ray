from typing import Union
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart
from specification.specification import MantaRayMorphologySpecification, MantaRayPectoralFinJointSpecification
import numpy as np

class MJCMantaRayPectoralFinSegment(MJCMorphologyPart):
    def __init__(self, 
                 parent: MJCMorphology | MJCMorphologyPart, 
                 name: str, 
                 pos: np.array, 
                 euler: np.array, 
                 side: int,
                 *args, **kwargs) -> None:
        self._side = side
        self._parent = parent
        super().__init__(parent, name, pos, euler, *args, **kwargs)
    
    @property
    def morphology_specification(self) -> MantaRayMorphologySpecification:
        return super().morphology_specification
    
    @property
    def center_of_capsule(self) -> np.ndarray:
        radius = self._pectoral_fin_segment_specification.radius.value
        length = self._pectoral_fin_segment_specification.length.value
        y_offset = radius + length
        return np.array([0, -self._side*y_offset, 0])
    
    def _build(self, segment_index: int) -> None:
        """
        side: if it is the right (dexter) pectoral fin, -1 if it is the left (sinister).
        """
        self._segment_index = segment_index
        self._pectoral_fin_specification = self.morphology_specification.pectoral_fin_specification
        self._pectoral_fin_segment_specification = self._pectoral_fin_specification.segment_specifications[segment_index]
        self._amount_of_segments = len(self._pectoral_fin_specification.segment_specifications)

        self._build_horizontal_part()
        self._configure_torso_joint()
        self._configure_tendon_site()
    
    def _build_horizontal_part(self) -> None:
        radius = self._pectoral_fin_segment_specification.radius.value
        length = self._pectoral_fin_segment_specification.length.value
        if self._side == 1:
            name = f"{self.base_name}_pectoral_right_fin_horizontal_{self._segment_index}"
        else:
            name = f"{self.base_name}_pectoral_left_fin_horizontal_{self._segment_index}"
        self.pectoral_backbone = self.mjcf_body.add(
                                "geom",
                                name=f"{self.base_name}_capsule",
                                type="capsule",
                                pos=self.center_of_capsule,
                                euler=[np.pi / 2, 0, 0],
                                size=[radius, length],
                                rgba=[100, 0, 0, 0.5], 
                                density=1000,
                                # fluidshape="ellipsoid",
                )
    
    def _configure_torso_joint(self) -> None:
        joint_specification = self._pectoral_fin_specification.joint_specification
        self.joint_x = self.mjcf_body.add("joint",
                           name=f"{self.base_name}_x_axis",
                            type="hinge",
                            axis=np.array([1, 0, 0]),
                            limited=True,
                            range=[-joint_specification.range.value, joint_specification.range.value],
                            damping=joint_specification.damping.value,
                            stiffness=joint_specification.stiffness.value,
                            )
        self.joint_z = self.mjcf_body.add("joint",
                           name=f"{self.base_name}_z_axis",
                            type="hinge",
                            axis=np.array([0, 0, 1]),
                            limited=True,
                            range=[-joint_specification.range.value/20, joint_specification.range.value/20],  # needs to be changed
                            damping=joint_specification.damping.value,
                            stiffness=joint_specification.stiffness.value,
                            )
    
    def _configure_tendon_site(self) -> None:
        self.tendon_site = self.mjcf_body.add("site",
                           name=f"{self.base_name}_tendon_site",
                            type="sphere",# visual representation of sit
                            size=0.0001*np.ones(3),
                            pos=self.center_of_capsule,
                            euler=np.zeros(3),
        )