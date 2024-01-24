import numpy as np
from typing import Union

from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart
from morphology.specification.specification import MantaRayMorphologySpecification, MantaRayTailJointSpecification

class MJCMantaRayTailSegment(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array, *args, **kwargs) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)
    
    @property
    def morphology_specification(self) -> MantaRayMorphologySpecification:
        return super().morphology_specification
    
    @property
    def center_of_capsule(
            self
            ) -> np.ndarray:
        radius = self._tail_segment_specification.radius.value
        length = self._tail_segment_specification.length.value
        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])
    
    
    
    def _build(self, segment_index: int,
               *args, **kwargs) -> None:
        self._segment_index = segment_index
        self._tail_specification = self.morphology_specification.tail_specification
        self._tail_segment_specification = self._tail_specification.segment_specifications[segment_index]
        self._actuator_specification = self.morphology_specification.actuation_specification

        self._build_capsule()
        self._configure_joints()
        self._configure_actuators()

    
    def _build_capsule(self) -> None:
        radius = self._tail_segment_specification.radius.value
        length = self._tail_segment_specification.length.value
        self.capsule =  self.mjcf_body.add(
                                "geom",
                                name=f"{self.base_name}_capsule",
                                type="capsule",
                                pos=self.center_of_capsule,
                                euler=[0, np.pi / 2, 0],
                                size=[radius, length / 2],
                                rgba=[100, 0, 0, 0.5], 
                                density=1000,
                                # fluidshape="ellipsoid",
                                )
    
    def _configure_joints(self) -> None:
        self.closest_joint = self._configure_joint(
            name=f"{self.base_name}_z_axis",
            axis=np.array([0, 0, 1]),   # left and right
            joint_specification=self._tail_segment_specification.joint_specification,
            )
        self.furthest_joint = self._configure_joint(
            name=f"{self.base_name}_y_axis",
            axis=np.array([0, 1, 0]),   # up and down
            joint_specification=self._tail_segment_specification.joint_specification,
        )
    
    def _configure_joint(self,
                         name: str,
                         axis: np.ndarray,
                         joint_specification: MantaRayTailJointSpecification,
                         ) -> mjcf.Element:
        joint = self.mjcf_body.add(
            "joint",
            name=name,
            type="hinge",
            limited=True,
            range=[-joint_specification.range.value, joint_specification.range.value],# see range for ball joint
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value,
        )
        return joint
    
    def _configure_actuators(self) -> None:
        self._configure_actuator(joint=self.closest_joint)
        self._configure_actuator(joint=self.furthest_joint)
    
    def _configure_actuator(self,
                            joint: mjcf.Element,
                            ) -> None:
        self.mjcf_model.actuator.add(
            "position",
            name=f"{joint.name}_actuator",
            joint=joint,
            kp=self._actuator_specification.kp.value,
            ctrllimited=True,
            ctrlrange=joint.range,
        )
    
    