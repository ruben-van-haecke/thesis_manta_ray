from __future__ import annotations
import sys
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray/')
from controller.parameters import MantaRayControllerSpecificationParameterizer
from controller.specification.default import default_controller_specification

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp
from typing import List
from morphology.morphology import MJCMantaRayMorphology

from morphology.specification.default import default_morphology_specification
from parameters import MantaRayMorphologySpecificationParameterizer
from task.drag_race import MoveConfig
from cmaes import CMA

from dm_control.mjcf import export_with_assets
from dm_control import viewer
from dm_env import TimeStep
from controller.specification.controller_specification import MantaRayCpgControllerSpecification

from gymnasium.core import ObsType
import time as time_module

class CPG():
    """Central Pattern Generator"""
    def __init__(self, 
                specification: MantaRayCpgControllerSpecification,
                low: float | None = -1,
                high: float | None = 1,
                ) -> None:
        """
        args:
            specification: MantaRayCpgControllerSpecification, the specification of the CPG
            low: float, lower bound of the output
            high: float, upper bound of the output
        """
        # self._neurons = [Neuron(specification=spec) for spec in specification.neuron_specifications]
        self._num_neurons = specification._num_neurons
        self._weights = specification.weights
        self._phase_biases = specification.phase_biases
        self._low = low
        self._high = high
        self._r = specification.r   #np.ones(shape=(self._num_neurons, ))
        self._x = specification.x   #np.zeros(shape=(self._num_neurons, ))
        self._omega = specification.omega   #np.ones(shape=(self._num_neurons, ))
        self._a_r = 20  # rad/s
        self._a_x = 20  # rad/s
        self._last_time = 0.
        self._last_state = np.zeros(shape=(5*self._num_neurons, ))
    
    def __str__(self):
        string = f"""number of oscillators: {self._num_neurons}
modulatie: 
    r: {self._r}
    omega: {self._omega}
    x: {self._x}

weights:
{self._weights}

phase biases:
{self._phase_biases}
"""
        return string
    
    
    
    def ask(self,
            observation: ObsType | None = None,
            duration: float | None = None,
            sampling_period: float | None = None,
            ) -> np.ndarray:
        """
        args:
            observation: ObsType, the observation of the environment, if None, then time is considered to be 0
            duration: float, duration of the integration, if None, only the value from the time_step is used
            sampling_period: float, time step of the integration, if None, only the value from the time_step is used
        returns:
            an (N, T) -dimensional vector, where N is the number of neurons, T is the number of time steps
        """
        # Define the differential equations that govern the dynamics
        def dynamics(t, y):
            """
            args:
                t: time, scalar
                y: state, np.ndarray [phi, r, x, r_dot, x_dot]
                phi, r, x: phase, amplitude, and offset of the oscillator
            """
            phi = y[0: self._num_neurons]
            r = y[1*self._num_neurons: 2*self._num_neurons]
            x = y[2*self._num_neurons: 3*self._num_neurons]
            r_dot = y[3*self._num_neurons: 4*self._num_neurons]
            x_dot = y[4*self._num_neurons: 5*self._num_neurons]

            out = np.empty(shape=(self._num_neurons*5, ))
            for neuron_index in range(self._num_neurons):
                out[neuron_index] = self._omega[0, neuron_index]+np.sum(self._weights[neuron_index, :]*self._r[0, :]*np.sin(phi-phi[neuron_index]-self._phase_biases[neuron_index, :]))  # phi_dot
            out[1*self._num_neurons: 2*self._num_neurons] = r_dot  # r_dot
            out[2*self._num_neurons: 3*self._num_neurons] = x_dot   # x_dot
            out[3*self._num_neurons: 4*self._num_neurons] = self._a_r * (self._a_r/4*(self._r[0, :]-r)-r_dot) # r_dotdot
            out[4*self._num_neurons: 5*self._num_neurons] = self._a_x * (self._a_x/4*(self._x[0, :]-x)-x_dot) # x_dotdot
            return out
        

        time = 0 if observation is None else observation["task/time"][0][0]
        np.random.seed(0)
        # Solve the differential equations using solve_ivp
        if time == 0:   # reset for new episode
            self._last_state = np.empty(shape=(5*self._num_neurons, ))
            self._last_state[0*self._num_neurons: 1*self._num_neurons] = np.sum(self._phase_biases[:, :], axis=0)-self._omega[0, :]#np.random.uniform(low=-2, high=2, size=self._num_neurons)
            self._last_state[4] = 0
            self._last_state[1*self._num_neurons: 2*self._num_neurons] = self._r[:]   # r
            self._last_state[2*self._num_neurons: 3*self._num_neurons] = self._x[:]  # x
            self._last_state[3*self._num_neurons: 4*self._num_neurons] = np.zeros(shape=(self._num_neurons,))  # r_dot
            self._last_state[4*self._num_neurons: 5*self._num_neurons] = np.zeros(shape=(self._num_neurons,))  # x_dot
            self._last_time = time
        if duration == None:    # only one time step
            if time == 0:
                return np.zeros(shape=(self._num_neurons, ))
            sol = solve_ivp(fun=dynamics, 
                            t_span=[self._last_time, time], 
                            y0=self._last_state,   # Initial conditions
                            t_eval=np.array([time]),
                            method="Radau",
                            )
            self._last_state = sol.y[:, -1]
            self._last_time = time
            out = sol.y[1*self._num_neurons:2*self._num_neurons, :] * np.cos(sol.y[0:self._num_neurons, :]) + sol.y[2*self._num_neurons:3*self._num_neurons, :]
        else:   # multiple time steps i.e. duration of 2 timesteps until the end of the episode
            sol = solve_ivp(fun=dynamics,
                            t_span=[self._last_time, time+duration],
                            y0=self._last_state,   # Initial conditions
                            t_eval=np.linspace(time, time+duration-sampling_period, int(np.ceil(duration/sampling_period))+1),
                            method='Radau',
                            )
            self._last_state = sol.y[:, -1]
            self._last_time = time
            out = sol.y[1*self._num_neurons:2*self._num_neurons, :] * np.cos(sol.y[0:self._num_neurons, :]) + sol.y[2*self._num_neurons:3*self._num_neurons, :]
        return out
    
    def tell(self):
        pass
        


# show a simple example of a CPG
if __name__ == "__main__":
    
    # morphology:
    morpholoby_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morpholoby_specification)
    manta_ray.export_to_xml_with_assets('morphology/manta_ray.xml') #just to be sure

    # task
    task_config = MoveConfig(simulation_time=6,
                             reward_fn='(E + 200*Δx) * (Δx)')
    dm_env = task_config.environment(morphology=manta_ray, wrap2gym=False)

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()
    export_with_assets(mjcf_model=dm_env.task.root_entity.mjcf_model, out_dir="morphology/manta_ray.xml")

    # controller
    controller_specification = default_controller_specification(action_spec=action_spec)
    action_spec = dm_env.action_spec()
    names = action_spec.name.split('\t')
    index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
    index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
    parameterizer = MantaRayControllerSpecificationParameterizer()
    parameterizer.parameterize_specification(specification=controller_specification)
    cpg = CPG(specification=controller_specification)

    timestep = dm_env.reset()
    # cpg_actions = cpg.ask(observation=timestep.observation,
    #                          duration=task_config.simulation_time,
    #                          sampling_period=task_config.control_timestep,
    #                          )
    cpg_modulations = []
    cpg_actions = []

    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec, cpg_actions, cpg_modulations, parameterizer, controller_specification
        names = action_spec.name.split('\t')
        time = timestep.observation["task/time"][0]

        num_actuators = action_spec.shape[0]
        actions = np.zeros(num_actuators)
        # fin_amplitude_left
        # fin_offset_left,
        # frequency_left 
        # phase_bias_left

        # fin_amplitude_right
        # fin_offset_right
        # frequency_right
        # phase_bias_right
        if time < 3:
            modulation = np.array([1., 0.5, 0.2, 0.,#left
                                   1., 0.5, 0.2, 1. # right
                                   ])
        else:
            modulation = np.array([1., 0.5, 0.2, 0.75, # left
                                   0.5, 0.25, 0.2, 0.25 # right
                                   ])
        parameterizer.parameter_space(specification=controller_specification,
                                      controller_action=modulation)
        actions = cpg.ask(observation=timestep.observation,
                        duration=None, #task_config.control_timestep,
                        sampling_period=task_config.control_timestep,
                        )
        if len(actions.shape) > 1:
            actions = actions[:, 0]
        # cpg_actions.append(actions)

        # rescale from [-1, 1] to actual joint range
        minimum, maximum = action_spec.minimum, action_spec.maximum

        normalised_actions = (actions + 1) / 2

        scaled_actions = minimum + normalised_actions * (maximum - minimum)
        cpg_actions.append(actions)
        cpg_modulations.append(parameterizer.get_scaled_parameters(specification=controller_specification))

        return scaled_actions


    viewer.launch(
            environment_loader=dm_env, 
            policy=oscillator_policy_fn
            )

    import plotly.graph_objects as go

    print(f"specification: {controller_specification}")

    # Extract the 4th and 6th elements from cpg_actions
    cpg_actions_4th = np.array([action[4] for action in cpg_actions])
    cpg_actions_6th = np.array([action[6] for action in cpg_actions])

    # Create a scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, task_config.simulation_time, len(cpg_actions_4th)-1), 
                             y=cpg_actions_4th[1:], 
                             name='oscillator 1'))
    fig.add_trace(go.Scatter(x=np.linspace(0, task_config.simulation_time, len(cpg_actions_6th)-1), 
                             y=cpg_actions_6th[1:], 
                             name='oscillator 2'))

    # Set plot layout
    fig.update_layout(title='CPG with transition',
                      xaxis_title='time [s]',
                      yaxis_title='oscillator output',
                      font=dict(size=25))

    # Show the plot
    fig.show()
    print(f"modulation 1: {cpg_modulations[0]}")
    print(f"modulation 2: {cpg_modulations[-1]}")