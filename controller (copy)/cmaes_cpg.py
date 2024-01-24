from __future__ import annotations
import sys
sys.path.insert(0,'/media/ruben/data/documents/unief/thesis/thesis_manta_ray/')
from controller.parameters import MantaRayControllerSpecificationParameterizer
from controller.specification.default import default_controller_dragrace_specification

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp
from typing import List
from morphology.morphology import MJCMantaRayMorphology

from morphology.specification.default import default_morphology_specification
from parameters import MantaRayMorphologySpecificationParameterizer
from task.drag_race import Move
from cmaes import CMA

from dm_control.mjcf import export_with_assets
from dm_control import viewer
from dm_env import TimeStep
from controller.specification.controller_specification import MantaRayCpgNeuronControllerSpecification, MantaRayCpgControllerSpecification

from gymnasium.core import ObsType
import time as time_module

class Neuron:
    time_used = 0.
    def __init__(self,
                 specification: MantaRayCpgNeuronControllerSpecification | None = None,
                 ) -> None:
        """
        args:
            specification: MantaRayCpgNeuronControllerSpecification, the specification of the neuron 
                if the specification is None then the R, X, omega are used for control and not fixed per episode for optimisation
        """
        self.history = []
        self.amplitude = []
        self.phase = []
        self._last_state = np.zeros(shape=(5, ))
        if specification is not None:
            self._omega = specification.omega
            self._R = specification.R
            self._X = specification.X
        self._a_r = 20  # rad/s
        self._a_x = 20  # rad/s

        self._last_time = 0
        self._last_state = np.zeros(shape=(5, ))

    def integrate(self, 
                  observation: ObsType,
                  w_ij: np.ndarray, # (to, from)
                  r_j: np.ndarray, 
                  phi_j: np.ndarray, 
                  small_phi_ij: np.ndarray,
                  duration: float | None = None,
                  sampling_period: float | None = None,
                  ):
        # Define the differential equations that govern the dynamics
        begin_time = time_module.time()
        def dynamics(t, y):
            """
            args:
                t: time, scalar
                y: state, np.ndarray [phi, r, x, r_dot, x_dot]
                phi, r, x: phase, amplitude, and offset of the oscillator
            """
            phi = y[0]
            r = y[1]
            x = y[2]
            r_dot = y[3]
            x_dot = y[4]
            return np.array([self._omega.value+np.sum(w_ij*r_j*np.sin(phi_j-phi-small_phi_ij)),  # phi_dot
                            r_dot,  # r_dot
                            x_dot,  # x_dot
                            self._a_r * (self._a_r/4*(self._R.value-r)-r_dot),   #r_dotdot
                            self._a_x * (self._a_x/4*(self._X.value-x)-x_dot),   #x_dotdot
                            ])
        time = observation["task/time"][0][0]
        # Solve the differential equations using solve_ivp
        if time == 0:
            self._last_state = np.array([0, 0, 0, 0, 0])
            self._last_time = time
            return self._last_state
        if duration == None:
            sol = solve_ivp(fun=dynamics, 
                            t_span=[self._last_time, time], 
                            y0=self._last_state,   # Initial conditions
                            t_eval=np.array([time]),
                            )
        else:
            sol = solve_ivp(fun=dynamics,
                            t_span=[0, duration],
                            y0=self._last_state,   # Initial conditions
                            t_eval=np.arange(0, duration, sampling_period),
                            )
        self._last_state = sol.y.flatten()
        self._last_time = time
        Neuron.time_used += time_module.time()-begin_time
        return sol.y.flatten()
    
    def get_neuron_respons(
            self, 
            observation: ObsType,
            w_ij: np.ndarray, 
            r_j: np.ndarray,
            phi_j: np.ndarray, 
            small_phi_ij: np.ndarray,
            duration: float | None = None,
            sampling_period: float | None = None,
            ) -> np.ndarray:
        """
        args:
            duration: float, duration of the integration
            time_step: float, time step of the integration
            w_ij: np.ndarray, weight matrix
            r_j: np.ndarray, amplitude of the other oscillators
            phi_j: np.ndarray, phase of the other oscillators
            small_phi_ij: np.ndarray, phase bias of the other oscillators
            duration: float, duration of the integration, if None, only the value from the time_step is used
            sampling_period: float, time step of the integration, if None, only the value from the time_step is used
        returns:
            (output, phase, amplitude, offset)
        """
        y = self.integrate(observation=observation,
                            w_ij=w_ij, 
                            r_j=r_j, 
                            phi_j=phi_j, 
                            small_phi_ij=small_phi_ij,
                            duration=duration,
                            sampling_period=sampling_period,
                            )
        if duration == None:
            time = observation["task/time"][0]
        else:
            time = np.arange(0, duration, sampling_period)
        out = y[1] * np.sin(y[0]*time + y[2]) 
        return out, y[0], y[1]

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
        self._neurons = [Neuron(specification=spec) for spec in specification.neuron_specifications]
        self._weights = specification.connections_specification.weight.value
        self._phase_biases = specification.connections_specification.phase_bias.value
        self._low = low
        self._high = high
        self._phi_old = np.ndarray(shape=(len(self._neurons), ))
        self._phi_new = np.ndarray(shape=(len(self._neurons), ))
        self._r_old = np.ndarray(shape=(len(self._neurons), ))
        self._r_new = np.ndarray(shape=(len(self._neurons), ))
            
    
    def ask(self,
            observation: ObsType,
            duration: float | None = None,
            sampling_period: float | None = None,
            ) -> np.ndarray:
        """
        args:
            observation: ObsType, the observation of the environment
            duration: float, duration of the integration, if None, only the value from the time_step is used
            sampling_period: float, time step of the integration, if None, only the value from the time_step is used
        returns:
            an (N, T) -dimensional vector, where N is the number of neurons i.e. one output value per time step
        """
        # time = time_step.observation["task/time"][0]
        solutions = np.ndarray(shape=(len(self._neurons), ))
        for index, neuron in enumerate(self._neurons):
            solutions[index: index+1], self._phi_new[index], self._r_new[index] = \
                            neuron.get_neuron_respons(observation=observation, 
                                                       w_ij=self._weights[index, :], 
                                                       r_j=self._r_old, 
                                                       phi_j=self._phi_old, 
                                                       small_phi_ij=self._phase_biases[index, :],
                                                       duration=duration,
                                                       sampling_period=sampling_period,)
        self._phi_old = self._phi_new
        self._r_old = self._r_new
        if self._low != None or self._high != None:
            return np.clip(solutions, self._low, self._high)
        else:
            return solutions
    
    def tell(self):
        pass
        


# show a simple example of a CPG
if __name__ == "__main__":
    
    # morphology:
    morpholoby_specification = default_morphology_specification()
    manta_ray = MJCMantaRayMorphology(specification=morpholoby_specification)
    manta_ray.export_to_xml_with_assets('morphology/manta_ray.xml') #just to be sure

    # task
    task_config = Move()
    dm_env = task_config.environment(morphology=manta_ray, wrap2gym=False)

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()
    export_with_assets(mjcf_model=dm_env.task.root_entity.mjcf_model, out_dir="morphology/manta_ray.xml")

    # controller
    controller_specification = default_controller_dragrace_specification()
    action_spec = dm_env.action_spec()
    names = action_spec.name.split('\t')
    index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
    index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
    parameterizer = MantaRayControllerSpecificationParameterizer(
        amplitude_fin_out_plane_range=(0, 1),
        frequency_fin_out_plane_range=(0, 3),
        offset_fin_out_plane_range=(-1, 1),
        left_fin_x=index_left_pectoral_fin_x,
        right_fin_x=index_right_pectoral_fin_x,
    )
    parameterizer.parameterize_specification(specification=controller_specification)
    MantaRayCpgNeuronControllerSpecification()
    cpg = CPG(specification=controller_specification)

    timestep = dm_env.reset()
    # cpg_actions = cpg.ask(observation=timestep.observation,
    #                          duration=task_config.simulation_time,
    #                          sampling_period=task_config.control_timestep,
    #                          )

    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec, cpg_actions
        names = action_spec.name.split('\t')
        time = timestep.observation["task/time"][0]
        index_left_pectoral_fin_x = names.index('morphology/left_pectoral_fin_actuator_x')
        index_right_pectoral_fin_x = names.index('morphology/right_pectoral_fin_actuator_x')
        index_left_pectoral_fin_z = names.index('morphology/left_pectoral_fin_actuator_z')
        index_right_pectoral_fin_z = names.index('morphology/right_pectoral_fin_actuator_z')

        num_actuators = action_spec.shape[0]
        actions = np.zeros(num_actuators)
        action = np.zeros(shape=(2, 1))
        column_index = int(time[0]/task_config.control_timestep)
        # action = cpg_actions[:, column_index]#.flatten()
        action = cpg.ask(observation=timestep.observation
                             )

        omega = 10
        left_fin_action_x = action[0]#np.cos(omega*time)
        left_fin_action_z = np.sin(omega*time)/20
        right_fin_action_x = action[1]#np.cos(omega*time+np.pi)
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
