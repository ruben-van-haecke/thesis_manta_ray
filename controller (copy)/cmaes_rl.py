from __future__ import annotations
import sys
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray/')

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
from task.move_to_target import Move
from morphology.specification.default import default_morphology_specification
from morphology.morphology import MJCMantaRayMorphology
# from cma import CMA
from cmaes import CMA
from parameters import MantaRayMorphologySpecificationParameterizer


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, 
                 obs_space_dims: int, 
                 action_space_dims: int, 
                 morphology_action: np.ndarray | None = None,
                 ):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
            morphology_action: The morphology parameters to be used for the MantaRayMorphologySpecification
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        print("state:", state)
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
    
    def run_episode(self,
                    morphology_action: np.ndarray) -> float:
        global gym_env, action_space
        done = False
        obs, _ = gym_env.reset()
        while not done:
            print("obs:", obs)
            action = self.sample_action(state=obs)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            self.rewards.append(reward)
            done = terminated or truncated
        self.update()
        return np.sum(self.rewards)


def run_generation(
        cma: CMA,
        agents: list[REINFORCE],
        ) -> (list[REINFORCE], list[tuple[float, float]]):
    solutions = []
    for agent in agents:
        morphology_action = cma.ask()
        reward = agent.run_episode(morphology_action=morphology_action)
        solutions.append((morphology_action, reward))
    return agents, solutions

if __name__ == "__main__":
    morphology_specification = default_morphology_specification()
    morphology = MJCMantaRayMorphology(specification=morphology_specification)
    parameterizer = MantaRayMorphologySpecificationParameterizer(
        torso_length_range=(0.05, 2.),
        torso_radius_range=(0.05, 2.),
        )
    parameterizer.parameterize_specification(specification=morphology_specification)

    task_config = Move(simulation_time=2,
                        target_distance_from_origin=0.5)
    gym_env = task_config.environment(morphology=morphology, 
                                        wrap2gym=True)
    action_space = gym_env.action_space
    morphology_space = parameterizer.get_target_parameters(specification=morphology_specification)
    print("morphology_space:", morphology_space)
    cma = CMA(mean=np.zeros(shape=len(morphology_space)),
              sigma=0.01,
              )
    # make the different agents/MantaRays
    agent = REINFORCE(obs_space_dims=1, 
                      action_space_dims=action_space.shape[0],
    )
    for generation in range(5):
        print("generation:", generation)
        agents, solutions = run_generation(cma=cma,
                                           agents=[agent])
        cma.tell(solutions)



    # CMA(initial_solution=np.zeros(shape=morphology_space.shape), 
    #                 initial_step_size=0.01,
    #                 fitness_functions=,  # time to reach target
    #                 # enforce_bounds=[],
    #                 population_size=10,
    #                 store_trace=True,
    #                 )