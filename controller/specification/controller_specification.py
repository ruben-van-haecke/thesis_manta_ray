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
            low: List[float] | np.ndarray = [],
            high: List[float] | np.ndarray = [],
            initial_value: np.ndarray | None = None, 
            ) -> None:
        """
        args:
            initial_value: instantiate the value on this array and do not make it modifieble.
        """
        super(NumpyArrayParameter, self).__init__(value=np.zeros(shape))
        self.shape = shape
        self._low: np.ndarray = np.array(low)
        self._high: np.ndarray = np.array(high)
        self._modifiable = []   # list of indices that can be modified
        if initial_value is not None:
            assert initial_value.shape == self.shape, f"[NumpyArrayParameter] Given initial_value shape '{initial_value.shape}' does not have shape '{len(self.shape)}'"
            self._value = initial_value

    @property
    def value(
            self
            ) -> float | np.ndarray:
        if self._value is None:
            self.set_random_value()

        # self._value = np.clip(self._value, self._low, self._high)
        return self._value[tuple(np.transpose(self._modifiable))]

    @value.setter
    def value(
            self,
            value: List[float] | np.ndarray
            ) -> None:
        if value.size == 0:
            return 
        try:
            value = np.array(value).reshape(self._value[tuple(np.transpose(self._modifiable))].shape)   # TODO: maybe not the cleanest way...
            self._value[tuple(np.transpose(self._modifiable))] = value
        except ValueError:
            raise ValueError(f"[NumpyArrayParameter] Given value '{value}' does not have length '{len(self._modifiable)}'")
    @property
    def low(
            self
            ) -> np.ndarray:
        return self._low
    @property
    def high(
            self
            ) -> np.ndarray:
        return self._high
        
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
        return f"NumpyArrayParameter(shape={self.shape}, low={self._low}, high={self._high}, value={self._value})"

    def set_random_value(
            self
            ) -> None:
        self._value = np.zeros(self.shape)
        self._value[tuple(np.transpose(self._modifiable))] = np.random.uniform(low=self._low, high=self._high, size=(len(self._modifiable),))
    
    def add_connections(self,
                        connections: List[tuple[int, int]],
                        low: List[float] | np.ndarray,
                        high: List[float] | np.ndarray,
                        weights: List[float] | np.ndarray | None = None,
                        ) -> None:
        for connection in connections:
            assert connection not in self._modifiable, f"[NumpyArrayParameter] Connection '{connection}' is already in the list of modifiable connections"
        
        self._modifiable += connections
        self._low = np.concatenate((self._low, low))
        self._high = np.concatenate((self._high, np.array(high)))
        if weights is not None:
            weights = np.array(weights).reshape(self._value[tuple(np.transpose(connections))].shape)    # TODO: maybe not the cleanest way...
            self._value[tuple(np.transpose(connections))] = weights

    def set_connections(self,
                       connections: List[tuple[int, int]],
                       weights: List[float] | np.ndarray,
                       ) -> None:
        """
        same as add_connections but fixes the connection to the given weight and can thus not be optimized
        """
        weights = np.array(weights).reshape(self._value[tuple(np.transpose(connections))].shape)    # TODO: maybe not the cleanest way...
        self._value[tuple(np.transpose(connections))] = weights

    

class MantaRayCpgControllerSpecification(ControllerSpecification):
    def __init__(self,
                 num_neurons: int,
                 action_spec: np.ndarray,
                 ) -> None:
        super().__init__()
        self._num_neurons = num_neurons
        self._action_spec = action_spec

        self.r = NumpyArrayParameter(shape=(1, self._num_neurons))
        self.x = NumpyArrayParameter(shape=(1, self._num_neurons))
        self.omega = NumpyArrayParameter(shape=(1, self._num_neurons))   # max 4 Hz
        
        self.weights = NumpyArrayParameter(shape=(self._num_neurons, self._num_neurons))
        self.phase_biases = NumpyArrayParameter(shape=(self._num_neurons, self._num_neurons))
    
    def __str__(self):
        overview = f"MantaRayCpgControllerSpecification:\n"
        overview += f"  num_neurons: {self._num_neurons}\n"
        overview += f"  action_spec: {self._action_spec}\n"
        overview += f"  r: {self.r.value}\n"
        overview += f"    low: {self.r.low}\n"
        overview += f"    high: {self.r.high}\n"
        overview += f"  x: {self.x.value}\n"
        overview += f"    low: {self.x.low}\n"
        overview += f"    high: {self.x.high}\n"
        overview += f"  omega:{self.omega.value}\n"
        overview += f"    low: {self.omega.low}\n"
        overview += f"    high: {self.omega.high}\n"
        overview += f"  weights: {self.weights.value}\n"
        overview += f"    low: {self.weights.low}\n"
        overview += f"    high: {self.weights.high}\n"
        overview += f"  phase_biases: {self.phase_biases.value}\n"
        overview += f"    low: {self.phase_biases.low}\n"
        overview += f"    high: {self.phase_biases.high}\n"
        return overview

    
    # def scaled_update(self,
    #                   update: np.ndarray,
    #                   ) -> None:
    #     """
    #         args:
    #             update: np.ndarray of shape (num_neurons, ) within range [0, 1]

    #         scales the update to the range of the parameter
    #     """
    #     assert np.all(update >= 0) and np.all(update <= 1), f"[MantaRayCpgControllerSpecification] Update '{update}' is not within range [0, 1]"
    #     # get the right length due to symmetry
    #     amplitude = update[0]
    #     offset = update[1]
    #     frequency = update[2]
    #     phase_bias = update[3]
        
    #     # updating specification
    #     self.r.value = self.r.low + amplitude * (self.r.high - self.r.low)

    #     self.x.value = self.x.low + offset * (self.x.high - self.x.low)

    #     self.omega.value = self.omega.low + frequency * (self.omega.high - self.omega.low)

    #     self.phase_biases.value = self.phase_biases.low + phase_bias * (self.phase_biases.high - self.phase_biases.low)
    #     print("scaled update =================================================")

    