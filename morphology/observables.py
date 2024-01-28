from typing import List

from dm_control import composer, mjcf
from dm_control.composer.observation.observable import MJCFFeature
from mujoco_utils.observables import ConfinedMJCFFeature

import numpy as np


class Observables(composer.Observables):
    sensor_actuatorfrc_names = []
    sensor_actuatorfrc = None

    @property
    def sencor_actuatorfrc(self) -> List[mjcf.Element]:
        if self.sensor_actuatorfrc is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self.sensor_actuatorfrc = list(filter(lambda sensor: sensor.tag == "actuatorfrc", sensors))
            
        return self.sensor_actuatorfrc
    
    @composer.observable
    def touch_per_tendon_plate(self) -> MJCFFeature:
        return ConfinedMJCFFeature(low=-np.inf,
                                   high=np.inf,
                                   shape=[len(self.sensor_actuatorfrc_names)],
                                   kind="sensordata",
                                   mjcf_element=self.sensor_actuatorfrc,
                                   )