import numpy as np

from morphology.specification.specification import MantaRayMorphologySpecification, \
MantaRayTorsoSpecification, MantaRayTailSegmentSpecification, \
MantaRayTailJointSpecification, MantaRayTailSpecification, MantaRayActuationSpecification,\
MantaRayPectoralFinSpecification, MantaRayPectoralFinJointSpecification, MantaRayPectoralFinSegmentSpecification,\
MantaRayPectoralFinTendonSpecification


scaling_factor = 1.0/22.5
# define default values
# tail
tail_length = 10*scaling_factor
tail_radius = 0.25*scaling_factor
num_tail_segments = 2
tail_segment_length = tail_length / num_tail_segments

# torso
torso_l = 5 * scaling_factor  # half-length of the torso
torso_r = 2.5 * scaling_factor  # radius of the torso

# pectoral fin
pectoral_fin_segment_radius = 0.45 * scaling_factor  # radius of the capsule according to x-axis
pectoral_fin_segment_length = 5 * scaling_factor # half-length according to the y-axis i.e. half the length of the fin
pectoral_fin_horizontal_segments = 10    #(in direction of x-axis)


def default_tail_joint_specification() -> MantaRayTailJointSpecification:
    return MantaRayTailJointSpecification(range=np.pi/4, # range is the range of motion of the joint expressed in radians
                                          stiffness=0.01,
                                          damping=1.)
def default_pectoral_fin_joint_specification() -> MantaRayPectoralFinJointSpecification:
    return MantaRayPectoralFinJointSpecification(range=np.pi/2, # range is the range of motion of the joint expressed in radians
                                                 stiffness=0.5,
                                                 damping=0.5)

def default_pectoral_fin_tendon_specification() -> MantaRayPectoralFinTendonSpecification:
    return MantaRayPectoralFinTendonSpecification(range_deviation=100,  # percentage that it can deviate from the initial length i.e. 1.0 is 100%
                                                  stiffness=1000,
                                                  damping=1)

def default_actuation_specification() -> MantaRayActuationSpecification:
    return MantaRayActuationSpecification(kp=100) 

def default_tail_segment_specification() -> MantaRayTailSegmentSpecification:
    return MantaRayTailSegmentSpecification(radius=tail_radius,
                                            length=tail_segment_length,
                                            joint_specifications=default_tail_joint_specification(),
                                            )
def default_tail_specification(num_segments: int,
                               ) -> MantaRayTailSpecification:
    
    return MantaRayTailSpecification(segment_specifications=[default_tail_segment_specification() for _ in range(num_segments)])

def default_pectoral_fin_segment_specification() -> MantaRayPectoralFinSegmentSpecification:
    return MantaRayPectoralFinSegmentSpecification(radius=pectoral_fin_segment_radius,
                                                   length=pectoral_fin_segment_length,
                                                    joint_specification=default_pectoral_fin_joint_specification(),
                                                    segment_index=0,
                                                    amount_of_segments=pectoral_fin_horizontal_segments,
                                                    )

def default_pectoral_fin_specification()->MantaRayPectoralFinSpecification:
    specs = []
    for segment_index in range(pectoral_fin_horizontal_segments):
        specs.append(
            MantaRayPectoralFinSegmentSpecification(
                radius=pectoral_fin_segment_radius,
                max_length=pectoral_fin_segment_length,
                joint_specification=default_pectoral_fin_joint_specification(),
                segment_index=segment_index,
                amount_of_segments=pectoral_fin_horizontal_segments,
                )
        )
    return MantaRayPectoralFinSpecification(segment_specifications=specs, 
                                            joint_specification=default_pectoral_fin_joint_specification(),
                                            tendon_specification=default_pectoral_fin_tendon_specification(),
                                            )

def default_torso_specification() -> MantaRayTorsoSpecification:
    return MantaRayTorsoSpecification(r_=torso_r,
                                      l_=torso_l
                                      )

def default_morphology_specification() -> MantaRayMorphologySpecification:
    return MantaRayMorphologySpecification(torso_specification=default_torso_specification(),
                                           tail_specification=default_tail_specification(num_segments=num_tail_segments),
                                           pectoral_fin_specification=default_pectoral_fin_specification(),
                                           actuation_specification=default_actuation_specification(),)