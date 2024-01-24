import sys

import numpy as np
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray/')
from typing import List

from fprs.parameters import ContinuousParameter
from fprs.specification_parameterizer import ControllerSpecificationParameterizer

from controller.specification.default import default_controller_dragrace_specification
from controller.specification.controller_specification import MantaRayCpgControllerSpecification


class MantaRayControllerSpecificationParameterizer(ControllerSpecificationParameterizer):
    def __init__(
            self,
            amplitude_fin_out_plane_range: tuple[float, float],
            frequency_fin_out_plane_range: tuple[float, float],
            offset_fin_out_plane_range: tuple[float, float],
            left_fin_x: int,
            right_fin_x: int
            ) -> None:
        super().__init__()
        self._amplitude_fin_out_plane_l_range = amplitude_fin_out_plane_range
        self._frequency_fin_out_plane_l_range = frequency_fin_out_plane_range
        self._offset_fin_out_plane_l_range = offset_fin_out_plane_range
        self._amplitude_fin_out_plane_r_range = amplitude_fin_out_plane_range
        self._frequency_fin_out_plane_r_range = frequency_fin_out_plane_range
        self._offset_fin_out_plane_r_range = offset_fin_out_plane_range

        self._left_fin_x = left_fin_x
        self._right_fin_x = right_fin_x

    def parameterize_specification(
            self,
            specification: MantaRayCpgControllerSpecification
            ) -> None:
        # Passing None as value will set the parameter to a random value within the given range
        ## fins
        # left fin
        specification.neuron_specifications[self._left_fin_x].R = ContinuousParameter(
                low=self._amplitude_fin_out_plane_l_range[0], high=self._amplitude_fin_out_plane_l_range[1], value=None
                )
        specification.neuron_specifications[self._left_fin_x].omega = ContinuousParameter(
                low=self._frequency_fin_out_plane_l_range[0], high=self._frequency_fin_out_plane_l_range[1], value=None
                )
        specification.neuron_specifications[self._left_fin_x].X = ContinuousParameter(
                low=self._offset_fin_out_plane_l_range[0], high=self._offset_fin_out_plane_l_range[1], value=None
                )
        # right fin
        specification.neuron_specifications[self._right_fin_x].R = ContinuousParameter(
                low=self._amplitude_fin_out_plane_r_range[0], high=self._amplitude_fin_out_plane_r_range[1], value=None
                )
        specification.neuron_specifications[self._right_fin_x].omega = ContinuousParameter(
                low=self._frequency_fin_out_plane_r_range[0], high=self._frequency_fin_out_plane_r_range[1], value=None
                )
        specification.neuron_specifications[self._right_fin_x].X = ContinuousParameter(
                low=self._offset_fin_out_plane_r_range[0], high=self._offset_fin_out_plane_r_range[1], value=None
                )
        # connections
        specification.connections_specification.add_connections(connections=[(self._left_fin_x, self._right_fin_x), (self._right_fin_x, self._left_fin_x)],
                                                               weights=[1, 1],
                                                               phase_biases=[0, 0])
        
        ## tail
        
        
    def parameter_space(self,
                        specification: MantaRayCpgControllerSpecification,
                        controller_action: np.ndarray,
                        ) -> None:
        specification.neuron_specifications[self._left_fin_x].R.value = controller_action[0]
        specification.neuron_specifications[self._left_fin_x].omega.value = controller_action[1]
        specification.neuron_specifications[self._left_fin_x].X.value = controller_action[2]
        specification.neuron_specifications[self._right_fin_x].R.value = controller_action[3]
        specification.neuron_specifications[self._right_fin_x].omega.value = controller_action[4]
        specification.neuron_specifications[self._right_fin_x].X.value = controller_action[5]
        specification.connections_specification.weight.value = controller_action[6:8]
        specification.connections_specification.phase_bias.value = controller_action[8:10]

    

    def get_parameter_labels(
            self,
            specification: MantaRayCpgControllerSpecification
            ) -> List[str]:
        return ["amplitude_fin_out_plane_l",
                "frequency_fin_out_plane_l",
                "offset_fin_out_plane_l",
                "amplitude_fin_out_plane_r",
                "frequency_fin_out_plane_r",
                "offset_fin_out_plane_r",
                "weight_left_right",
                "weight_right_left",
                "phase_bias_left_right",
                "phase_bias_right_left",
                ]



if __name__ == '__main__':
    controller_specification = default_controller_dragrace_specification()

    parameterizer = MantaRayControllerSpecificationParameterizer(
        amplitude_fin_out_plane_range=(0, 1),
        frequency_fin_out_plane_range=(0, 3),
        offset_fin_out_plane_range=(-1, 1),
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