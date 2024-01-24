
from controller.specification.controller_specification import MantaRayCpgControllerSpecification, MantaRayCpgNeuronControllerSpecification


def default_controller_dragrace_specification() -> MantaRayCpgControllerSpecification:
    return MantaRayCpgControllerSpecification([     # order isn't right -> action_spec.name.split('\t') is used to get the right order
        MantaRayCpgNeuronControllerSpecification(), # left fin out of plane
        MantaRayCpgNeuronControllerSpecification(), # right fin out of plane
        MantaRayCpgNeuronControllerSpecification(), # left fin in plane
        MantaRayCpgNeuronControllerSpecification(), # right fin in plane
        MantaRayCpgNeuronControllerSpecification(), # segment 1 tail out of plane
        MantaRayCpgNeuronControllerSpecification(), # segment 1 tail in plane
        MantaRayCpgNeuronControllerSpecification(), # segment 2 tail out of plane
        MantaRayCpgNeuronControllerSpecification(), # segment 2 tail in plane
    ])