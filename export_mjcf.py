import sys
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray')

import mujoco
from PIL import Image
import matplotlib.pyplot as plt

from morphology.morphology import MJCMantaRayMorphology
from morphology.specification.specification import MantaRayMorphologySpecification
from morphology.specification.default import default_morphology_specification

if __name__ == "__main__":
    morpholoby_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morpholoby_specification)
    manta_ray.export_to_xml_with_assets('morphology/manta_ray.xml')
