from dm_control import composer, mjcf

class CilinderParkour(composer.Entity):
    def _build(
            self
            ):
        self._mjcf_model = mjcf.RootElement()
        self._geom = self._mjcf_model.worldbody.add(
                'geom', type='cylinder', 
                size=[0.5, 1], 
                rgba=(100, 100, 100, 0.5),
                )

    @property
    def mjcf_model(
            self
            ):
        return self._mjcf_model