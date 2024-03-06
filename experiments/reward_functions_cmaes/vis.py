import sys
import os

import sys
import os

import numpy as np

from task.drag_race import DragRaceTask

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from evolution_simulation_correct_version import OptimizerSimulation

# Load the .pkl file
this_dir = "/media/ruben/data/documents/unief/thesis/thesis_manta_ray/experiments/reward_functions_cmaes/"

def get_sim(velocity: float, 
            reward_fn: str,
            ) -> OptimizerSimulation:
    success = False
    while not success:
        try:
            velocity_str = "{:.2f}".format(velocity)
            reward_str = reward_fn.replace("*", "_times_")
            sim = OptimizerSimulation.load("reward_functions_cmaes/sim_objects/energy_velocity_"+velocity_str+"_reward_function_"+reward_str)
            success = True
        except Exception as e:
            print(e)
            # if not os.path.isfile("energy_velocity_experiment/sim_objects/energy_velocity_"+velocity_str+"_reward_function_"+reward_str+".pkl"):
            #     print("file does not exist")
            #     break
            print("Could not instantiate the .pkl file, 2 reasons:\n \
                    1) make sure that the class has not changed since the last save\n \
                    2) instantiating the environment in the class might fail, it retries and should solve itself \n")
            print("file: ", "reward_functions_cmaes/sim_objects/energy_velocity_"+velocity_str+"_reward_function_"+reward_str)
    return sim


def compare_all(save: bool=True) -> go.Figure:
    # Plot the results
    fig = go.Figure()

    for reward_fn in DragRaceTask.reward_functions:
        velocity = 0.7
        sim: OptimizerSimulation = get_sim(velocity, reward_fn)
        avg_gen = np.average(sim._outer_rewards, axis=1)
        fig.add_trace(go.Scatter(x=list(range(len(avg_gen))), y=avg_gen, name=f"{reward_fn}_{velocity}"))

    fig.update_layout(
        title="Average Reward per Generation",
        xaxis_title="Generation",
        yaxis_title="Avg Inverse Reward"
    )

    fig.show()
    if save: pio.write_html(fig, this_dir + f"plots/compare_all.html", full_html=False)
    return fig

def scatter_sim(velocity: float,
                reward_fn: str,
                save: bool=True
                ) -> go.Figure:
    sim: OptimizerSimulation = get_sim(velocity, reward_fn)
    fig = go.Figure()
    for episode in range(sim._outer_rewards.shape[1]):
        fig.add_trace(go.Scatter(x=list(range(len(sim._outer_rewards[:, episode]))), 
                                y=sim._outer_rewards[:, episode], 
                                name=f"episode: {episode}",
                                mode='markers'
                                ))
    fig.update_layout(
        title=f"Velocity: {velocity}, Inverse Reward Function: {reward_fn}",
        xaxis_title="Generation",
        yaxis_title="Inverse Reward"
    )
    fig.show()
    if save: pio.write_html(fig, this_dir + f"plots/sim_vel_{velocity}_reward_{reward_fn}.html", full_html=False)
    return fig

def plot_actions(velocity: float,
                 reward_fn: str,
                 gen: int,
                 episode: int,
                 save: bool=True
                 ) -> go.Figure:
    sim: OptimizerSimulation = get_sim(velocity, reward_fn)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(sim._actions[gen, episode, 4, :]))), 
                            y=sim._actions[gen, episode, 4, :], 
                            name=f"right fin",
                            ))
    fig.add_trace(go.Scatter(x=list(range(len(sim._actions[gen, episode, 6, :]))), 
                            y=sim._actions[gen, episode, 6, :], 
                            name=f"left fin",
                            ))
    fig.show()
    if save: pio.write_html(fig, this_dir + f"plots/actions_{velocity}_reward_{reward_fn}_gen_{gen}_episode_{episode}.html", full_html=False)
    return fig



print("Terminal")
while True:
    inp = input("    >>> ")
    if inp=="exit" or input=="quit" or input=="exit()" or input=="quit()":
        break
    try:
        exec(inp)
    except Exception as e:
        print(e)
        continue


