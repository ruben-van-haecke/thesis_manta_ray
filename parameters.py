from typing import List

from fprs.parameters import ContinuousParameter
from fprs.specification_parameterizer import MorphologySpecificationParameterizer

from morphology.specification.default import default_morphology_specification
from morphology.specification.specification import MantaRayMorphologySpecification


class MantaRayMorphologySpecificationParameterizer(MorphologySpecificationParameterizer):
    def __init__(
            self,
            torso_length_range: tuple[float, float],
            torso_radius_range: tuple[float, float],
            ) -> None:
        super().__init__()
        self._torso_length_range = torso_length_range
        self._torso_radius_range = torso_radius_range

    def parameterize_specification(
            self,
            specification: MantaRayMorphologySpecification
            ) -> None:
        # Passing None as value will set the parameter to a random value within the given range
        specification.torso_specification.radius = ContinuousParameter(
                low=self._torso_radius_range[0], high=self._torso_radius_range[1], value=None
                )
        specification.torso_specification.length = ContinuousParameter(
                low=self._torso_length_range[0], high=self._torso_length_range[1], value=None
                )

    def get_parameter_labels(
            self,
            specification: MantaRayMorphologySpecification
            ) -> List[str]:
        return ["torso-radius", "(half) torso-length)"]


if __name__ == '__main__':
    morphology_specification = default_morphology_specification()

    parameterizer = MantaRayMorphologySpecificationParameterizer(
        torso_length_range=(0.01, 2.),
        torso_radius_range=(0.01, 2.),
    )
    parameterizer.parameterize_specification(specification=morphology_specification)

    print("All parameters:")
    print(f"\t{morphology_specification.parameters}")
    print()
    print("Parameters to optimise:")
    for parameter, label in zip(
            parameterizer.get_target_parameters(specification=morphology_specification),
            parameterizer.get_parameter_labels(specification=morphology_specification)
            ):
        print(f"\t{label}\t->\t{parameter}")