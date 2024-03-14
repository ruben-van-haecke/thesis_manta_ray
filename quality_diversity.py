import pickle
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
    
    @behaviour.setter
    def behaviour(self, value):
        self._behaviour = value
    
    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def fitness(self, value):
        self._fitness = value
    
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    def __lt__(self, other: 'Solution') -> bool:
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return f"behaviour: {self._behaviour}, fitness: {self._fitness}, parameters: {self._parameters}"
    
    def copy(self) -> 'Solution':
        """
        Create a copy of the solution.

        :return: A copy of the solution.
        """
        return Solution(behaviour=self._behaviour.copy(),
                        fitness=self._fitness,
                        parameters=self._parameters.copy())



class Archive:
    def __init__(self, 
                 parameter_bounds: List[Tuple[float, float]],
                 feature_bounds: List[Tuple[float, float]], 
                 resolutions: np.ndarray | List[int], 
                 parameter_names: List[str],
                 feature_names: List[str],
                 symmetry: List[Tuple[str, str]] = [],
                 max_items_per_bin: int = np.inf,
                 ) -> None:
        """
        Initialize the MAP-Elites archive.
        
        :param parameter_bounds: A list of tuples indicating the min and max values for each parameter.
        :param feature_bounds: A list of tuples indicating the min and max values for each feature. This is the same as the behaviour space.
        :param resolutions: The number of bins along each dimension of the feature space.
        :param parameter_names: A list of strings indicating the name of each parameter.
        :param feature_names: A list of strings indicating the name of each feature.
        :param symmetry: if empty List, no symmetrical features are considered. 
                        Otherwise, the BD get flipped around 0 and the parameters are flipped according to the tuples in the list
        :param max_items_per_bin: The maximum number of items to store in each bin. If None, there is no limit.
        """
        assert len(feature_bounds) == len(resolutions), "The number of feature bounds must match the number of resolutions."
        assert len(parameter_bounds) == len(parameter_names), "The number of parameter bounds must match the number of parameter names."
        assert len(feature_bounds) == len(feature_names), "The number of feature bounds must match the number of feature names."
        self._parameter_names = parameter_names
        self._feature_names = feature_names 

        self._feature_dimensions = len(feature_bounds)
        self._parameter_bounds = parameter_bounds
        self._feature_bounds = feature_bounds
        self._resolutions = resolutions
        self._symmetry = symmetry
        self._max_items_per_bin = max_items_per_bin

        self._number_of_bins = np.prod(self._resolutions)
        self._solutions: Dict[Tuple[int, ...], List[Solution]] = {}
        for combination in itertools.product(*[range(res) for res in self._resolutions]):
            self._solutions[tuple(combination)] = []
    @property
    def max_items_per_bin(self) -> int:
        return self._max_items_per_bin
    
    @property
    def solutions(self) -> Dict[Tuple[int, ...], List[Solution]]:
        return self._solutions
    
    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        return self._parameter_bounds

    def min_per_bin(self) -> Dict[Tuple[int, ...], float]:
        """
        Returns the minimum fitness of solutions per bin
        """
        min_fitness = {}
        for index, solutions in self._solutions.items():
            if len(solutions) > 0:
                min_fitness[index] = min([sol.fitness for sol in solutions])
            else:
                min_fitness[index] = 0
        return min_fitness
    
    def avg_per_bin(self) -> Dict[Tuple[int, ...], float]:
        """
        Returns the average fitness of solutions per bin
        """
        avg_fitness = {}
        for index, solutions in self._solutions.items():
            if len(solutions) > 0:
                avg_fitness[index] = np.mean([sol.fitness for sol in solutions])
            else:
                avg_fitness[index] = 0
        return avg_fitness
    
    def max_per_bin(self) -> Dict[Tuple[int, ...], float]:
        """
        Returns the maximum fitness of solutions per bin
        """
        max_fitness = {}
        for index, solutions in self._solutions.items():
            if len(solutions) > 0:
                max_fitness[index] = max([sol.fitness for sol in solutions])
            else:
                max_fitness[index] = 0
        return max_fitness
    
    def num_sol_per_bin(self) -> Dict[Tuple[int, ...], int]:
        """
        Returns the number of solutions per bin
        """
        num_sol = {}
        for index, solutions in self._solutions.items():
            num_sol[index] = len(solutions)
        return num_sol
    
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
    
    
    def get_symmetric_solution(self, 
                                sol: Solution,
                                ) -> Solution:
        """
        returns a list of all the symmetrical Solutions, including the original solution
        if no symmetrical features are present, returns a list with the original solution
        """
        new_sol: Solution = sol.copy()
        new_sol.behaviour = -sol.behaviour
        for parameter1, parameter2 in self._symmetry:
            parameter1_index = self._parameter_names.index(parameter1)
            parameter2_index = self._parameter_names.index(parameter2)
            new_sol.parameters[parameter1_index], new_sol.parameters[parameter2_index] = \
                sol.parameters[parameter2_index], new_sol.parameters[parameter1_index]
        return new_sol
    

    def __iter__(self):
        for solutions in self._solutions.values():
            yield from solutions
    
    def __len__(self):
        return sum(len(solutions) for solutions in self._solutions.values())

    def add_solution(self, solution: Solution):
        all_solutions = [solution]
        # get the other solution
        other_solution = self.get_symmetric_solution(solution)
        all_solutions.append(other_solution)

        # Add all solutions to the archive
        for sol in all_solutions:
            index = self.get_bin_index(sol.behaviour)
            self._solutions[index].append(sol)
            if len(self._solutions[index]) > self._max_items_per_bin:
                self._solutions[index].remove(min(self._solutions[index]))

    
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
        index_bin_data = []
        fitness_data = []
        index_data = []

        for sol in self:
            x_data.append(sol.behaviour[x_index])
            y_data.append(sol.behaviour[y_index])
            index_bin_data.append(self.get_bin_index(sol.behaviour))
            fitness_data.append(sol.fitness)
            index_data.append(self._solutions[self.get_bin_index(sol.behaviour)].index(sol))

        fig = go.Figure()
        df = pd.DataFrame({'x': x_data, 'y': y_data, 'index_bin': index_bin_data, 'fitness': fitness_data, 'index': index_data})

        # Add scatter plot for data points
        px_fig = px.scatter(df, x='x', y='y', hover_data=df.columns, color='fitness')
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
    

    def get_bins(self) -> List[Tuple[int, ...]]:
        return list(self._solutions.keys())
    
    def get_best_solution(self, index: Tuple[int, ...]) -> Solution | None:
        """
        Returns the best solution in the bin corresponding to the index
        """
        solutions = self._solutions[index]
        if len(solutions) == 0:
            return None
        return max(solutions, key=lambda sol: sol.fitness)

    def store(self, filename: str):
        """
        Store the archive to a file.

        :param filename: The name of the file to store the archive.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(filename: str) -> 'Archive':
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    



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
        if len(self._archive) < 30:
            return np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
        else:
            np.random.seed()    # otherwise the same random number is generated every time -> why?
            ## selection
            # Make probabilities features for choosing a bin
            bins = self._archive.get_bins()
            bins_max_fitness = np.array([self._archive.max_per_bin()[bin] for bin in bins])
            bins_min_fitness = np.array([self._archive.min_per_bin()[bin] for bin in bins])
            bins_avg_fitness = np.array([self._archive.avg_per_bin()[bin] for bin in bins])
            bins_num_sol = np.array([self._archive.num_sol_per_bin()[bin] for bin in bins])
            # normalize
            bins_max_fitness = bins_max_fitness/np.sum(bins_max_fitness)
            bins_min_fitness = bins_min_fitness/np.sum(bins_min_fitness)
            bins_avg_fitness = bins_avg_fitness/np.sum(bins_avg_fitness)
            bins_num_sol = bins_num_sol/np.max(bins_num_sol)
            probs = 80*(1-bins_num_sol) + 20*(1-bins_max_fitness)
            probs = probs/np.sum(probs)
            # choose a bin
            bin_index = np.random.choice(range(len(bins)), p=probs)
            bin_index = bins[bin_index]
            bin: List[Solution] = self._archive.solutions[bin_index]
            # choose a 2 solutions as parents
            if len(bin) == 0:
                parent1 = np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
                parent2 = np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
            elif len(bin) == 1:
                parent1 = bin[0].parameters
                parent2 = np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
            else:   # 2 or more
                fitnesses = np.array([sol.fitness for sol in bin])
                fitnesses_normalised = fitnesses/np.sum(fitnesses)
                bin_indices = np.random.choice(range(len(bin)), 2, p=fitnesses_normalised)
                parent1 = bin[bin_indices[0]].parameters
                parent2 = bin[bin_indices[1]].parameters

            ## crossover
            par_to_change = np.random.choice(range(len(parent1)), int(len(parent1)/2), [1/len(parent1) for _ in range(len(parent1))])
            child = parent1.copy()
            child[par_to_change] = parent2[par_to_change]

            ## mutation
            child += np.random.normal(0.5, 0.05, len(child))
            child = np.clip(child, 0, 1)
            return child


    def tell(self, solutions: List[Solution]):
        """
        Update the grid with a new solution.

        :param solutions: a list of solutions to add to the archive.
        """
        for sol in solutions:
            index = self._archive.add_solution(solution=sol)
        
        self._archive.store("archive.pkl")


if __name__ == "__main__":
    archive = Archive(feature_bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)], resolutions=[2, 2, 1], feature_names=["pitch", "yawn", "roll"])
    me = MapElites(archive)
    archive.plot_grid(x_label="pitch", y_label="roll")