import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology
from thesis_manta_ray.morphology.specification.specification import MantaRayMorphologySpecification
from thesis_manta_ray.morphology.specification.default import default_morphology_specification

from task.parkour import Move
# from task.grid_target import Move
# from task.move_to_target import Move
# from task.drag_race import Move

from dm_control import mjcf
from dm_control.mjcf import export_with_assets



if __name__ == "__main__":
    morpholoby_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morpholoby_specification)
    manta_ray.export_to_xml_with_assets('morphology/manta_ray.xml') #just to be sure

    task_config = Move()
    dm_env = task_config.environment(morphology=manta_ray, wrap2gym=False)

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()
    export_with_assets(mjcf_model=dm_env.task.root_entity.mjcf_model, out_dir="morphology/manta_ray.xml")

    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec
        names = action_spec.name.split('\t')
        time = timestep.observation["task/time"][0]
        index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
        index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
        index_left_pectoral_fin_z = names.index('morphology/left_pectoral_fin_actuator_z')
        index_right_pectoral_fin_z = names.index('morphology/right_pectoral_fin_actuator_z')

        num_actuators = action_spec.shape[0]
        actions = np.zeros(num_actuators)

        omega = 10
        left_fin_action_x = np.cos(omega*time)
        left_fin_action_z = np.sin(omega*time)/20
        right_fin_action_x = np.cos(omega*time+np.pi)
        right_fin_action_z = np.sin(omega*time+np.pi)/20

        actions[index_left_pectoral_fin_x:index_left_pectoral_fin_x+1] = left_fin_action_x
        actions[index_right_pectoral_fin_x:index_right_pectoral_fin_x+1] = right_fin_action_x
        actions[index_left_pectoral_fin_z:index_left_pectoral_fin_z+1] = left_fin_action_z
        actions[index_right_pectoral_fin_z:index_right_pectoral_fin_z+1] = right_fin_action_z
        # actions[1::2] = out_of_plane_actions

        # rescale from [-1, 1] to actual joint range
        minimum, maximum = action_spec.minimum, action_spec.maximum

        normalised_actions = (actions + 1) / 2

        scaled_actions = minimum + normalised_actions * (maximum - minimum)

        return scaled_actions


    viewer.launch(
            environment_loader=dm_env, 
            policy=oscillator_policy_fn
            )