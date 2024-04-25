import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from controller.cmaes_cpg_vectorized import CPG
from controller.parameters import MantaRayControllerSpecificationParameterizer
from controller.quality_diversity import Archive
from controller.specification.default import default_controller_specification
from task.bezier_parkour import BezierParkour
from thesis_manta_ray.morphology.morphology import MJCMantaRayMorphology
from thesis_manta_ray.morphology.specification.specification import MantaRayMorphologySpecification
from thesis_manta_ray.morphology.specification.default import default_morphology_specification
from thesis_manta_ray.controller.rule_based import translate_rotate, rotate

from task.drag_race import MoveConfig
# from task.grid_target import Move
# from task.move_to_target import Move
# from task.drag_race import Move

from dm_control import mjcf
from dm_control.mjcf import export_with_assets
from scipy.spatial.transform import Rotation


if __name__ == "__main__":
    morphology_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morphology_specification)
    manta_ray.export_to_xml_with_assets('morphology/manta_ray.xml') #just to be sure

    parkour = BezierParkour.load("task/parkours/slight_curve.pkl")
    task_config = MoveConfig(velocity=0.5, 
                             simulation_time=30,
                       reward_fn="(E + 200*Δx) * (Δx)",
                       task_mode="random_target",
                       )
    task_config.target_location = np.array([-5, 2, 1.5])
    dm_env = task_config.environment(morphology=MJCMantaRayMorphology(specification=morphology_specification), wrap2gym=False)
    # controller
    action_spec = dm_env.action_spec()
    controller_specification = default_controller_specification(action_spec=action_spec)
    controller_parameterizer = MantaRayControllerSpecificationParameterizer(
    )
    controller_parameterizer.parameterize_specification(specification=controller_specification)
    cpg = CPG(specification=controller_specification,
                low=-1,
                high=1,
                )
    archive: Archive = Archive.load("experiments/qd_v0.5_differential/sim_objects/archive.pkl")

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()
    export_with_assets(mjcf_model=dm_env.task.root_entity.mjcf_model, out_dir="morphology/manta_ray.xml")
    rolling_features = np.zeros(3)

    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec, task_config, archive, controller_specification, controller_parameterizer, rolling_features
        names = action_spec.name.split('\t')
        obs = timestep.observation
        current_angular_positions=obs["task/orientation"][0]
        current_xyz_velocities=obs["task/xyz_velocity"][0]
        current_position=obs["task/position"][0]
        #check if the transformation is correct
        target_location_after_transformation = translate_rotate(point=task_config.target_location,
                         translatation=current_position,
                         rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        fish_location_after_transformation = translate_rotate(point=current_position,
                         translatation=current_position,
                         rotation=-current_angular_positions+np.array([0, 0, np.pi]))
        # find the behaviour descriptor
        # roll
        roll = 0
        # pitch
        pitch = np.arctan(np.abs(target_location_after_transformation[2]/target_location_after_transformation[0]))
        pitch += np.pi/2 if target_location_after_transformation[0] < 0 else 0
        if target_location_after_transformation[2] < 0: # negative pitch (there has been a rotation of 180 degrees around the z-axis)
            pitch *= -1
        # yaw
        yaw = np.arctan(np.abs(target_location_after_transformation[1]/target_location_after_transformation[0]))
        yaw += np.pi/2 if target_location_after_transformation[0] < 0 else 0
        if target_location_after_transformation[1] < 0: # negative yaw
            yaw *= -1

        # scaling
        m = max(np.abs(target_location_after_transformation[2]), 
                   np.linalg.norm(np.array([target_location_after_transformation[0], target_location_after_transformation[1]])))
        pitch = np.abs(target_location_after_transformation[2])/m * pitch
        yaw = np.linalg.norm(np.array([target_location_after_transformation[0], target_location_after_transformation[1]]))/m * yaw

        features = np.array([roll, pitch, yaw]).reshape(1, -1)
        rolling_features = rolling_features * 0.9 + features * 0.1

        # get parameters
        sol = archive.get_closest_solutions(feature=rolling_features, k=1)[0][0]
        controller_parameterizer.parameter_space(specification=controller_specification,
                                             controller_action=sol.parameters)
        print_flag = False
        if print_flag:
            print(f"------------------------------------")
            print(f"angular_position: {current_angular_positions}")
            print(f"target_location_after_transformation: {target_location_after_transformation}")
            print(f"fish_location_after_transformation: {fish_location_after_transformation}")
            print(f"features: {features}")

        time = timestep.observation["task/time"][0]
        index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
        index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
        index_left_pectoral_fin_z = names.index('morphology/left_pectoral_fin_actuator_z')
        index_right_pectoral_fin_z = names.index('morphology/right_pectoral_fin_actuator_z')
        # actuation
        minimum, maximum = action_spec.minimum, action_spec.maximum
        if True:
            normalised_action = (cpg.ask(observation=timestep.observation,
                                            duration=None,  # one time step
                                            sampling_period=task_config.physics_timestep
                                            )+1)/2
            scaled_actions = minimum + normalised_action * (maximum - minimum)
            if len(scaled_actions.shape) == 2:
                scaled_actions = scaled_actions[:, 0]
        else:
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

            normalised_actions = (actions + 1) / 2

            scaled_actions = minimum + normalised_actions * (maximum - minimum)

        return scaled_actions


    viewer.launch(
            environment_loader=dm_env, 
            policy=oscillator_policy_fn
            )