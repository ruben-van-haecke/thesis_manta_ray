import numpy as np
from typing import List, Optional
from fprs.parameters import FixedParameter, ContinuousParameter, Parameter
from fprs.specification import ControllerSpecification, Specification

# create np.ndarray parameter
class NumpyArrayParameter(Parameter):
    """Inherits from Parameter, the .value attribute can be modified for optimization. Otherwise, use as np.ndarray"""
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
        self._modifiable = []   # list of indices that can be modified

    @property
    def value(
            self
            ) -> float:
        if self._value is None:
            self.set_random_value()

        self._value = np.clip(self._value, self.low, self.high)
        return self._value#[tuple(np.transpose(self._modifiable))]

    @value.setter
    def value(
            self,
            value: List[float] | np.ndarray
            ) -> None:
        try:
            value = np.array(value).reshape(self._value[tuple(np.transpose(self._modifiable))].shape)   # TODO: maybe not the cleanest way...
            self._value[tuple(np.transpose(self._modifiable))] = value
        except ValueError:
            raise ValueError(f"[NumpyArrayParameter] Given value '{value}' does not have length '{len(self._modifiable)}'")
        
    def __getitem__(self, index):
        return self._value[index]
    def __setitem__(self, index, value):
        # Check if index is in modifiable
        if index in self._modifiable:
            self._value[index] = value
        else:
            raise ValueError(f"Index {index} is not modifiable")  
    def __mul__(self, other)-> np.ndarray:
        if isinstance(other, (int, float, np.ndarray)):
            return self._value * other
        else:
            raise TypeError(f"Unsupported operand type for *: '{type(other).__name__}'")
    def __rmul__(self, other)-> np.ndarray:
        if isinstance(other, (int, float, np.ndarray)):
            return other * self._value
        else:
            raise TypeError(f"Unsupported operand type for *: '{type(other).__name__}'")
    def __add__(self, other)->np.ndarray:
        if isinstance(other, (int, float, np.ndarray)):
            return self._value + other
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other).__name__}'")
    def __sub__(self, other)-> np.ndarray:
        if isinstance(other, (int, float, np.ndarray)):
            return self._value - other
        else:
            raise TypeError(f"Unsupported operand type for -: '{type(other).__name__}'")
    def __str__(self):
        return str(self._value)
    def __repr__(self):
        return f"NumpyArrayParameter(shape={self.shape}, low={self.low}, high={self.high}, value={self._value})"

    def set_random_value(
            self
            ) -> None:
        self._value = np.zeros(self.shape)
        self._value[tuple(np.transpose(self._modifiable))] = np.random.uniform(low=self.low, high=self.high, size=(len(self._modifiable),))
    
    def add_connections(self,
                        connections: List[tuple[int, int]],
                        weights: List[float] | np.ndarray | None = None,
                        ) -> None:
        for connection in connections:
            assert connection not in self._modifiable, f"[NumpyArrayParameter] Connection '{connection}' is already in the list of modifiable connections"
        self._modifiable += connections
        if weights is not None:
            weights = np.array(weights).reshape(self._value[tuple(np.transpose(connections))].shape)    # TODO: maybe not the cleanest way...
            self._value[tuple(np.transpose(connections))] = weights

    

class MantaRayCpgControllerSpecification(ControllerSpecification):
    def __init__(self,
                 #neuron_specifications: List[MantaRayCpgNeuronControllerSpecification],
                 num_neurons: int,
                 ) -> None:
        super().__init__()
        self._num_neurons = num_neurons
        self.r = NumpyArrayParameter(shape=(1, self._num_neurons))
        self.x = NumpyArrayParameter(shape=(1, self._num_neurons))
        self.omega = NumpyArrayParameter(shape=(1, self._num_neurons))
        self.weights = NumpyArrayParameter(shape=(self._num_neurons, self._num_neurons), low=0, high=10)
        self.phase_biases = NumpyArrayParameter(shape=(self._num_neurons, self._num_neurons), low=-np.pi, high=np.pi)
    