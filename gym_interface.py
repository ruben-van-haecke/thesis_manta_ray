from morphology.specification.default import default_morphology_specification
from morphology.morphology import MJCMantaRayMorphology
from task.move_to_target import Move
from utils.video import show_video

from cma import CMA
import numpy as np
import tensorflow as tf



tf.random.set_seed(444)  # set random seed for reproducibility

if __name__ == "__main__":
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)

    task_config = Move(simulation_time=2,
                       target_distance_from_origin=0.5)
    gym_env = task_config.environment(morphology=morphology, 
                                      wrap2gym=True)
    action_space = gym_env.action_space

    # cma = CMA(initial_solution=np.zeros(shape=action_space.shape), 
    #                 initial_step_size=0.01,
    #                 fitness_functions=run_episode,  # time to reach target
    #                 # enforce_bounds=[],
    #                 population_size=10,
    #                 store_trace=True,
    #                 )

    def run_episode():
        global gym_env, action_space
        for generation in range(10):
            print("generation:", generation)
            done = False
            obs, _ = gym_env.reset()
            solutions = []
            while not done:
                action = action_space.sample()
                # action = optimizer.ask()
                obs, reward, terminated, truncated, info = gym_env.step(action)
                solutions.append((action, reward))
                done = terminated or truncated
                yield gym_env.render(camera_ids=[0])
            # optimizer.tell(solutions)

    # best_solution, best_fitness = cma.search()

    show_video(frame_generator=run_episode())
