from fprs.specification import MorphologySpecification
from thesis_manta_ray.morphology.specification.specification import MantaRayActuationSpecification,\
      MantaRayMorphologySpecification, MantaRayTailSpecification, MantaRayTorsoSpecification,\
        MantaRayPectoralFinJointSpecification, MantaRayPectoralFinSpecification
from typing import Union
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart
from thesis_manta_ray.morphology.parts.pectoral_fin_segment import MJCMantaRayPectoralFinSegment
import numpy as np


class MJCMantaRayPectoralFin(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], 
                 name: str, pos: np.array, euler: np.array, side: int, *args, **kwargs) -> None:
        self._side = side
        super().__init__(parent, name, pos, euler, *args, **kwargs)
    
    @property
    def morphology_specification(self) -> MorphologySpecification:
        return super().morphology_specification
    
    def _build(self) -> None:
        """
        side: 1 if it is the right (dexter) pectoral fin, -1 if it is the left (sinister).
        """
        self._pectoral_fin_specification: MantaRayPectoralFinSpecification = self.morphology_specification.pectoral_fin_specification
        self._tendon_specification: MantaRayPectoralFinJointSpecification = self._pectoral_fin_specification.tendon_specification
        self._build_segments()
        self._configure_tendons()
        # self._configure_flex()
        self._configure_actuator()
        self._configure_force_sensor()
    
    def _build_segments(self) -> None:
        self._segments = []
        self._number_of_segments = len(self._pectoral_fin_specification.segment_specifications)
        pos = 0
        for segment_index in range(self._number_of_segments):
            # try:
            #     parent = self._segments[-1]
            #     position = 2 * self._segments[-1].center_of_capsule
            #     if self._side == 1: # right
            #         pos = self._parent.pectoral_right_fin_position
            #     else:   # left
            #         pos = self._parent.pectoral_left_fin_position
            # except IndexError:  # first segment
            #     if self._side == 1:
            #         pos = self._parent.pectoral_right_fin_position
            #         name = f"{self.base_name}_pectoral_right_fin"
            #     else:
            #         pos = self._parent.pectoral_left_fin_position
            #         name = f"{self.base_name}_pectoral_left_fin"

            # spread the horizontal segments uniform along a side of the torso from the pectoral_fin_position
            # along a length of the length_of_fin 
            pos += np.array([self._parent.length_of_fin/(self._number_of_segments), 0, 0])
            segment = MJCMantaRayPectoralFinSegment(
                    parent=self,
                    name=f"{self.base_name}_segment_{segment_index}",
                    pos=pos,
                    euler=np.zeros(3),
                    segment_index=segment_index,
                    side=self._side,
                    )
            # self.mjcf_model.option.flag.contact = 'disable' # disable contact between segments
            self._segments.append(segment)
    
    def _configure_tendons(self) -> None:
        tendon_name = f"{self.base_name}_tendon_right" if self._side == 1 else f"{self.base_name}_tendon_left"
        normal_length_tendon = np.linalg.norm(self._segments[0].center_of_capsule - self._segments[-1].center_of_capsule)
        deviation = self._tendon_specification.range_deviation.value   # 5 percent
        self._tendon = self.mjcf_model.tendon.add("spatial", 
                                                name=tendon_name,
                                                limited=True,
                                                range=[normal_length_tendon*(1-deviation), normal_length_tendon*(1+deviation)],
                                                width=0.005,
                                                stiffness=self._tendon_specification.stiffness.value,
                                                damping=self._tendon_specification.damping.value,
                                                )
        for segment in self._segments:
            site_name = segment.tendon_site.name
            self._tendon.add("site", site=site_name)
    
    def _configure_actuator(self) -> None:
        self.mjcf_model.actuator.add(
            "position",
            name=f"{self.base_name}_actuator_x",
            joint=self._segments[0].joint_x.name,
            kp=self.morphology_specification.actuation_specification.kp.value,
            ctrllimited=True,
            ctrlrange=[-self._pectoral_fin_specification.joint_specification.range.value, self._pectoral_fin_specification.joint_specification.range.value],
        )
        self.mjcf_model.actuator.add(
            "position",
            name=f"{self.base_name}_actuator_z",
            joint=self._segments[0].joint_z.name,
            kp=self.morphology_specification.actuation_specification.kp.value,
            ctrllimited=True,
            ctrlrange=[-self._pectoral_fin_specification.joint_specification.range.value, self._pectoral_fin_specification.joint_specification.range.value],
        )
    
    def _configure_force_sensor(self) -> None:
        self.mjcf_model.sensor.add(
            "actuatorfrc",
            name=f"{self.base_name}_force_sensor_x",
            actuator=f"{self.base_name}_actuator_x",
        )
        self.mjcf_model.sensor.add(
            "actuatorfrc",
            name=f"{self.base_name}_force_sensor_z",
            actuator=f"{self.base_name}_actuator_z",
        )
    
    # def _configure_flex(self) -> None:
    #     body_names = ""
    #     for segment in self._segments:
    #         body_names += f"{segment._name} "
        
    #     self.mjcf_model.deformable.add(
    #         "flex",
    #         name=f"{self.base_name}_flex",
    #         body=body_names,
    #     )
            