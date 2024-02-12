
from controller.specification.controller_specification import MantaRayCpgControllerSpecification


def default_controller_dragrace_specification(action_spec) -> MantaRayCpgControllerSpecification:
    return MantaRayCpgControllerSpecification(num_neurons=8, action_spec=action_spec)

