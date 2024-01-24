import numpy as np
from dm_control import mjcf
from dm_control.composer import Entity

from utils import colors


class Obstacle(Entity):
    @property
    def mjcf_model(
            self
            ):
        return self._mjcf_model

    def _build(
            self,
            size: np.ndarray,
            *args,
            **kwargs
            ) -> None:
        self._mjcf_model = mjcf.RootElement(model="obstacle")
        self.mjcf_model.worldbody.add(
                'geom', type='cylinder', size=size, rgba=colors.rgba_gray,
                pos=[0.0, 0.0, size[-1]],
                contype=1,
                conaffinity=1
                )
