import numpy as np
from controller.specification.controller_specification import MantaRayCpgControllerSpecification

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class CPG():
    """Central Pattern Generator"""
    def __init__(self, 
                specification: MantaRayCpgControllerSpecification,
                ) -> None:
        """
        args:
            specification: MantaRayCpgControllerSpecification, the specification of the CPG
            low: float, lower bound of the output
            high: float, upper bound of the output
        """
        self._num_neurons = specification._num_neurons
        self._weights = specification.weights
        self._phase_biases = specification.phase_biases
        self._r = specification.r   #np.ones(shape=(self._num_neurons, ))
        self._x = specification.x   #np.zeros(shape=(self._num_neurons, ))
        self._omega = specification.omega   #np.ones(shape=(self._num_neurons, ))
        self._a_r = 20  # rad/s
        self._a_x = 20  # rad/s
    
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
            duration: float | None = None,
            sampling_period: float | None = None,
            ) -> np.ndarray:
        """
        args:
            observation: ObsType, the observation of the environment
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
                # self._omega and self._r are both of type NumpyArrayParameter, defined in controller/specification/controller_specification.py and comes down to a numpy array as indexed below
                out[neuron_index] = self._omega[0, neuron_index]+np.sum(self._weights[neuron_index, :]*self._r[0, :]*np.sin(phi-phi[neuron_index]-self._phase_biases[neuron_index, :]))  # phi_dot
            out[1*self._num_neurons: 2*self._num_neurons] = r_dot  # r_dot
            out[2*self._num_neurons: 3*self._num_neurons] = x_dot   # x_dot
            out[3*self._num_neurons: 4*self._num_neurons] = self._a_r * (self._a_r/4*(self._r[0, :]-r)-r_dot) # r_dotdot
            out[4*self._num_neurons: 5*self._num_neurons] = self._a_x * (self._a_x/4*(self._x[0, :]-x)-x_dot) # x_dotdot
            return out
        

        # set initial conditions for Differential Equation
        np.random.seed(0)
        self._last_state = np.empty(shape=(5*self._num_neurons, ))
        self._last_state[0*self._num_neurons: 1*self._num_neurons] = np.random.uniform(low=0, high=0.2, size=self._num_neurons)
        self._last_state[1*self._num_neurons: 2*self._num_neurons] = self._r[:]   # r
        self._last_state[2*self._num_neurons: 3*self._num_neurons] = self._x[:]  # x
        self._last_state[3*self._num_neurons: 4*self._num_neurons] = np.zeros(shape=(self._num_neurons,))  # r_dot
        self._last_state[4*self._num_neurons: 5*self._num_neurons] = np.zeros(shape=(self._num_neurons,))  # x_dot
        sol = solve_ivp(fun=dynamics,
                        t_span=[0, duration],
                        y0=self._last_state,   # Initial conditions
                        t_eval=np.linspace(0, duration, int(duration/sampling_period)),
                        )
        out = sol.y[1*self._num_neurons:2*self._num_neurons, :] * np.cos(sol.y[0:self._num_neurons, :]) + sol.y[2*self._num_neurons:3*self._num_neurons, :]
        return out

if __name__ == "__main__":
    controller_spec = MantaRayCpgControllerSpecification(num_neurons=2)
    simulation_time = 20
    timestep = 0.002

    omega = 2   # in Hz
    bias = np.pi
    controller_spec.r.add_connections(connections=[(0, 0), (0, 1)],
                                        weights=[1, 1.])
    controller_spec.omega.add_connections(connections=[(0, 0), (0, 1)],
                                        weights=np.ones(shape=(2, ))*np.pi*2*omega)
    connections = [(0, 1), (1, 0)]
    controller_spec.weights.add_connections(connections=connections,
                                            weights=[1, 1.],)
    controller_spec.phase_biases.add_connections(connections=connections,
                                            weights=[-bias, bias])
    
    cpg = CPG(specification=controller_spec)
    print(cpg)
    all_actions = np.empty(shape=(2, int(simulation_time/timestep)))
    all_actions = cpg.ask(duration=20,
                            sampling_period=0.002)
    time = np.arange(0, simulation_time, timestep)

    plt.plot(time, all_actions[1, :], label="left fin")
    plt.plot(time, all_actions[0, :], label="right fin")
    plt.legend()
    plt.show()
