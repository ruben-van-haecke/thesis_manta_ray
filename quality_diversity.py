import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
import plotly.graph_objects as go
import itertools
import plotly.io as pio
import plotly.express as px

class Solution:
    def __init__(self,
                 behaviour: np.ndarray,
                 fitness: float,
                 parameters: np.ndarray,
                 ):
        """
        Solution to be used in Archive

        :param behaviour: behavioural descriptor
        :param fitness: fitness to minimize
        :param parameters: parameters to optimize, in the case of the manta-ray thesis -> CPG parameters
        """
        self._behaviour = behaviour
        self._fitness = fitness
        self._parameters = parameters
    
    @property
    def behaviour(self):
        return self._behaviour
    @property
    def fitness(self):
        return self._fitness
    @property
    def parameters(self):
        return self._parameters
    
    def __str__(self) -> str:
        return f"behaviour: {self._behaviour}, fitness: {self._fitness}, parameters: {self._parameters}"



class Archive:
    def __init__(self, 
                 parameter_bounds: List[Tuple[float, float]],
                 feature_bounds: List[Tuple[float, float]], 
                 resolutions: np.ndarray | List[int], 
                 parameter_names: None | List[str] = None,
                 feature_names: None | List[str] = None,
                 max_items_per_bin: int | None = None,
                 ) -> None:
        """
        Initialize the MAP-Elites archive.
        
        :param parameter_bounds: A list of tuples indicating the min and max values for each parameter.
        :param feature_bounds: A list of tuples indicating the min and max values for each feature. This is the same as the behaviour space.
        :param resolutions: The number of bins along each dimension of the feature space.
        :param parameter_names: A list of strings indicating the name of each parameter.
        :param feature_names: A list of strings indicating the name of each feature.
        :param max_items_per_bin: The maximum number of items to store in each bin. If None, there is no limit.
        """
        assert len(feature_bounds) == len(resolutions), "The number of feature bounds must match the number of resolutions."
        if parameter_names is not None:
            assert len(parameter_bounds) == len(parameter_names), "The number of parameter bounds must match the number of parameter names."
        if feature_names is not None:
            assert len(feature_bounds) == len(feature_names), "The number of feature bounds must match the number of feature names."
        self._parameter_names = parameter_names
        self._feature_names = feature_names 

        self._feature_dimensions = len(feature_bounds)
        self._parameter_bounds = parameter_bounds
        self._feature_bounds = feature_bounds
        self._resolutions = resolutions

        self._number_of_bins = np.prod(self._resolutions)
        self._solutions: Dict[Tuple[int, ...], List[Solution]] = {}
        for combination in itertools.product(*[range(res) for res in self._resolutions]):
            self._solutions[tuple(combination)] = []
    
    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return self._parameter_bounds

    def __str__(self) -> str:
        return " ".join(str(sol) for sol in self)
    
    def get_bin_index(self, features: List[float]) -> Tuple[int, ...]:
        """
        returns a tuple with the indices of the bin corresponding to the features
        """
        index = []
        for res, bound, feature in zip(self._resolutions, self._feature_bounds, features):
            comparator = np.linspace(bound[0], bound[1], res+1)
            index.append(np.argmax(feature < comparator)-1)
        return tuple(index)
    
    def __iter__(self):
        for solutions in self._solutions.values():
            yield from solutions
    
    def __len__(self):
        return sum(len(solutions) for solutions in self._solutions.values())

    def add_solution(self, solution: Solution):
        index = self.get_bin_index(solution.behaviour)
        self._solutions[index].append(solution)
    
    def plot_grid(self,
                  x_label: str,
                  y_label: str,
                  title: str = "MAP-Elites Archive",
                  store: str | None = None,
                  ) -> None:
        """
        Plot the MAP-Elites archive.

        :param x_label: The label for the x-axis.
        :param y_label: The label for the y-axis.
        :param title: The title of the plot.
        :param store: The directory and filename to store the plot as an HTML file. If None, the plot is not stored
        """
        assert x_label in self._feature_names, "The x_label must be one of the feature names."
        assert y_label in self._feature_names, "The y_label must be one of the feature names."

        x_index = self._feature_names.index(x_label)
        y_index = self._feature_names.index(y_label)
        x_data = []
        y_data = []
        index_data = []
        fitness_data = []

        for sol in self:
            x_data.append(sol.behaviour[x_index])
            y_data.append(sol.behaviour[y_index])
            index_data.append(self.get_bin_index(sol.behaviour))
            fitness_data.append(sol.fitness)
        fig = go.Figure()
        df = pd.DataFrame({'x': x_data, 'y': y_data, 'index': index_data, 'fitness': fitness_data})

        # Add scatter plot for data points
        px_fig = px.scatter(df, x='x', y='y', color='fitness')
        for trace in px_fig.data:
            fig.add_trace(trace)

        # Add vertical lines
        x_vertical = np.linspace(self._feature_bounds[x_index][0], 
                        self._feature_bounds[x_index][1], 
                        self._resolutions[x_index] + 1)
        for x in list(x_vertical):
            fig.add_shape(type='line', 
                          x0=x, y0=self._feature_bounds[y_index][0], 
                          x1=x, y1=self._feature_bounds[y_index][1],
                        line=dict(color='RoyalBlue', width=1))

        # Add horizontal lines
        y_horizontal = np.linspace(self._feature_bounds[y_index][0], 
                        self._feature_bounds[y_index][1], 
                        self._resolutions[y_index] + 1)
        for y in y_horizontal:
            fig.add_shape(type='line', 
                          x0=self._feature_bounds[x_index][0], y0=y, 
                          x1=self._feature_bounds[x_index][1], y1=y,
                        line=dict(color='RoyalBlue', width=1))

        # # Set axes ranges
        # fig.update_xaxes(range=[0, max(x_data) + 1])
        # fig.update_yaxes(range=[0, max(y_data) + 1])

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
        )
        if store is not None: pio.write_html(fig, store, full_html=False)

            # Show figure
        fig.show()
    



class MapElites:
    def __init__(self, 
                 archive: Archive,
                 ) -> None:
        """
        Initialize the MAP-Elites algorithm.

        :param archive: The archive to use.
        """
        self._archive = archive
    
    @property
    def archive(self):
        return self._archive


    def ask(self) -> np.ndarray:
        """
        Generates a new parameter sample
        """
        if len(self._archive) == 0:
            return np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
        else:
            parameters = np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
            # selection
            # mutation
            # crossover
            return np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])


    def tell(self, solutions: List[Solution]):
        """
        Update the grid with a new solution.

        :param solutions: a list of solutions to add to the archive.
        """
        print(f"Adding {len(solutions)} solutions to the archive")
        for sol in solutions:
            index = self._archive.add_solution(solution=sol)


if __name__ == "__main__":
    archive = Archive(feature_bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)], resolutions=[2, 2, 1], feature_names=["pitch", "yawn", "roll"])
    me = MapElites(archive)
    archive.plot_grid(x_label="pitch", y_label="roll")