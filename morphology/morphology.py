import numpy as np
from mujoco_utils.robot import MJCMorphology

from thesis_manta_ray.morphology.parts.torso import MJCMantaRayTorso
from thesis_manta_ray.morphology.parts.tail import MJCMantaRayTail
from thesis_manta_ray.morphology.parts.pectoral_fin import MJCMantaRayPectoralFin
from thesis_manta_ray.morphology.specification.specification import MantaRayMorphologySpecification




class MJCMantaRayMorphology(MJCMorphology):
    def __init__(self, specification: MantaRayMorphologySpecification) -> None:
        super().__init__(specification)
        self._number_of_horizontal_fin_segments = 3

    @property
    def number_of_horizontal_fin_segments(self,) -> int:
        return self._number_of_horizontal_fin_segments
    
    @property
    def morphology_specification(self) -> MantaRayMorphologySpecification:
        return super().morphology_specification
    
    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = "radian"
    
    def _build(self, *args, **kwargs) -> None:
        self._configure_compiler()

        self._build_torso()
        self._build_tail()
        self._build_pectoral_fins()
    
    def _build_torso(self) -> None:
        self._torso = MJCMantaRayTorso(parent=self,
                                       name="torso",
                                       pos=np.zeros(3),
                                       euler=np.zeros(3),
                                       )
    def _build_tail(self) -> None:
        tail_pos = self._torso.tail_position
        self._tail = MJCMantaRayTail(parent=self,
                                     name="tail",
                                     pos=tail_pos,#np.zeros(3),
                                     euler=np.zeros(3),
                                     )
    def _build_pectoral_fins(self) -> None:
        self._right_fin = MJCMantaRayPectoralFin(parent=self._torso,
                                                 name="right_pectoral_fin",
                                                 pos=self._torso.pectoral_right_fin_position,
                                                 euler=np.zeros(3),
                                                 side=1)
        self._left_fin = MJCMantaRayPectoralFin(parent=self._torso,
                                                name="left_pectoral_fin",
                                                pos=self._torso.pectoral_left_fin_position,
                                                euler=np.zeros(3),
                                                side=-1)