import numpy as np
from typing import List, Optional
from fprs.parameters import FixedParameter, ContinuousParameter, Parameter
from fprs.specification import ControllerSpecification, Specification

# create np.ndarray parameter
class NumpyArrayParameter(Parameter):
    def __init__(
            self,
            shape: tuple[int, int],
            low: float = -1.0,
            high: float = 1.0,
            ) -> None:
        super(NumpyArrayParameter, self).__init__(value=np.zeros(shape))
        self.shape = shape
        self.low = low
        self.high = high
        self._mofifiable = []   # list of indices that can be modified

    @property
    def value(
            self
            ) -> float:
        if self._value is None:
            self.set_random_value()

        self._value = np.clip(self._value, self.low, self.high)
        return self._value

    @value.setter
    def value(
            self,
            value
            ) -> None:
        try:
            self._value[tuple(np.transpose(self._mofifiable))] = value
        except ValueError:
            raise ValueError(f"[NumpyArrayParameter] Given value '{value}' does not have length '{len(self._mofifiable)}'")

    def set_random_value(
            self
            ) -> None:
        self._value = np.zeros(self.shape)
        self._value[tuple(np.transpose(self._mofifiable))] = np.random.uniform(low=self.low, high=self.high, size=(len(self._mofifiable),))
    
    def add_connections(self,
                        connections: List[tuple[int, int]],
                        weights: List[float],
                        ) -> None:
        for connection in connections:
            assert connection[0] != connection[1], f"[NumpyArrayParameter] Connection '{connection}' is not allowed to be self-connected"
            assert connection not in self._mofifiable, f"[NumpyArrayParameter] Connection '{connection}' is already in the list of modifiable connections"
        self._mofifiable += connections
        self._value[tuple(np.transpose(connections))] = weights



class MantaRayCpgNeuronControllerSpecification(ControllerSpecification):
    def __init__(self,
                R: float = 0,
                X: float = 0,
                omega: float = 0,
                 ) -> None:
        super().__init__()
        self.R = FixedParameter(value=R)        # amplitude 
        self.X = FixedParameter(value=X)        # offset
        self.omega = FixedParameter(value=omega)    # frequency

class CpgNeuronConnectionSpecification(Specification):
    def __init__(self,
                 shape: tuple[int, int],
                 ) -> None:
        super().__init__()
        self.weight = NumpyArrayParameter(shape=shape)
        self.phase_bias = NumpyArrayParameter(shape=shape)
    
    def add_connections(self,
                        connections: List[tuple[int, int]],
                        weights: List[float],
                        phase_biases: List[float],
                        ) -> None:
        """
        makes the connections modifiable

        :param connections: list of tuples of connected neurons (to, from)
        :param weights: list of weights for each connection
        :param phase_biases: list of phase biases for each connection"""
        self.weight.add_connections(connections=connections, weights=weights)
        self.phase_bias.add_connections(connections=connections, weights=phase_biases)
    

class MantaRayCpgControllerSpecification(ControllerSpecification):
    def __init__(self,
                 neuron_specifications: List[MantaRayCpgNeuronControllerSpecification],
                 ) -> None:
        super().__init__()
        self.neuron_specifications = neuron_specifications
        amount_neurons = len(self.neuron_specifications)
        self.connections_specification = CpgNeuronConnectionSpecification(shape=(amount_neurons, amount_neurons))
        # self.weights = NumpyArrayParameter(shape=(amount_neurons, amount_neurons))
        # self.phase_biases = NumpyArrayParameter(shape=(amount_neurons, amount_neurons))
    