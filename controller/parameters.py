import sys

import numpy as np
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray/')
from typing import List

from fprs.parameters import ContinuousParameter
from fprs.specification_parameterizer import ControllerSpecificationParameterizer

from controller.specification.default import default_controller_specification
from controller.specification.controller_specification import MantaRayCpgControllerSpecification


class MantaRayControllerSpecificationParameterizer(ControllerSpecificationParameterizer):
    tail_segment_0_z = 0
    tail_segment_0_y = 1
    tail_segment_1_z = 2
    tail_segment_1_y = 3
    right_fin_x = 4
    right_fin_z = 5
    left_fin_x = 6
    left_fin_z = 7
    def __init__(self,) -> None:
        super().__init__()

    def parameterize_specification(
            self,
            specification: MantaRayCpgControllerSpecification
            ) -> None:
        omega = 4
        bias = np.pi
        specification.r.add_connections(connections=[(0, self.right_fin_x), 
                                                     (0, self.left_fin_x),
                                                     ],
                                        weights=[1, 1.],
                                        low=[0, 0], 
                                        high=[1, 1],
                                        )
        specification.x.add_connections(connections=[(0, self.right_fin_x), 
                                                     (0, self.left_fin_x),
                                                     ],
                                        weights=[1, 1.],
                                        low=[-np.pi/2, -np.pi/2], 
                                        high=[np.pi/2, np.pi/2],
                                        )
        specification.omega.add_connections(connections=[(0, self.right_fin_x), 
                                                         (0, self.left_fin_x),
                                                         ],
                                                weights=np.ones(shape=(2, ))*np.pi*2*omega,
                                                low=np.zeros(shape=(2, )),
                                                high=np.ones(shape=(2, ))*2*np.pi*omega)
        connections = [(4, 6), (6, 4),] # connection right-left fin
        specification.weights.set_connections(connections=connections,
                                                weights=[5, 5],
                                                )
        specification.phase_biases.add_connections(connections=connections,
                                                weights=[-bias, bias], 
                                                low=-np.ones(shape=(2, ))*np.pi,
                                                high=np.ones(shape=(2, ))*np.pi)
        
        
    def parameter_space(self,
                        specification: MantaRayCpgControllerSpecification,
                        controller_action: np.ndarray,
                        ) -> None:
        """
            args:
                controller_action: np.ndarray of shape (num_neurons, ) within range [0, 1]

            scales the controller_action to the range of the parameter
        """
        assert np.all(controller_action >= 0) and np.all(controller_action <= 1), f"[MantaRayCpgControllerSpecification] controller_action '{controller_action}' is not within range [0, 1]"
        # get the right length due to symmetry
        try:
            amplitude = controller_action[[0, 4]]    # left, right
        except IndexError:
            print(controller_action)
        offset = controller_action[[1, 5]]
        frequency = controller_action[[2, 6]]
        phase_bias = controller_action[[3, 7]]
        phase_bias[1] = 1-phase_bias[0]  # loops should sum to a multiple of 2*pi, in this case sum to 0
        
        # updating specification
        specification.x.value = -1 + offset*(2)#specification.x.low + offset * (specification.x.high - specification.x.low) # bounded in -1, 1
        specification.r.value = np.minimum(specification.r.low + amplitude * (specification.r.high - specification.r.low), # bounded in -1, 1
                                           1-np.abs(specification.x.value))

        specification.omega.value = specification.omega.low + frequency * (specification.omega.high - specification.omega.low)
        specification.phase_biases.value = specification.phase_biases.low + phase_bias * (specification.phase_biases.high - specification.phase_biases.low)
    
    def get_scaled_parameters(self,
                              specification: MantaRayCpgControllerSpecification,
                              ) -> np.ndarray:
        scaled_action = np.zeros(shape=(8, ))
        scaled_action[[0, 4]] = specification.r.value
        scaled_action[[1, 5]] = specification.x.value
        scaled_action[[2, 6]] = specification.omega.value
        scaled_action[[3, 7]] = specification.phase_biases.value
        return scaled_action
    
    

    def get_parameter_labels(
            self,
            ) -> List[str]:
        return ["fin_amplitude_left", "fin_offset_left", "frequency_left", "phase_bias_left",
                "fin_amplitude_right", "fin_offset_right", "frequency_right", "phase_bias_right"]



if __name__ == '__main__':
    controller_specification = default_controller_specification()

    parameterizer = MantaRayControllerSpecificationParameterizer(
        amplitude_fin_out_plane_range=(0, 1),
        frequency_fin_out_plane_range=(0, 3),
        offset_fin_out_plane_range=(-1, 1),
        left_fin_x=4,
        right_fin_x=6
    )
    parameterizer.parameterize_specification(specification=controller_specification)

    print("All parameters:")
    print(f"\t{controller_specification.parameters}")
    print()
    print("Parameters to optimise:")
    for parameter, label in zip(
            parameterizer.get_target_parameters(specification=controller_specification),
            parameterizer.get_parameter_labels(specification=controller_specification)
            ):
        print(f"\t{label}\t->\t{parameter}")