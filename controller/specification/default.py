
from controller.specification.controller_specification import MantaRayCpgControllerSpecification


def default_controller_dragrace_specification() -> MantaRayCpgControllerSpecification:
    return MantaRayCpgControllerSpecification(num_neurons=8)