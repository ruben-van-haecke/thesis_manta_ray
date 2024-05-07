import pickle
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
import plotly.graph_objects as go
import itertools
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
import plotly.io as pio
import plotly.express as px

class Solution:
    def __init__(self,
                 behaviour: np.ndarray,
                 fitness: float,
                 parameters: np.ndarray,
                 metadata: Dict = {},
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
        self._metadata = metadata
    
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
    
    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, value):
        assert isinstance(value, dict), "metadata must be a dictionary"
        self._metadata = value

    def __lt__(self, other: 'Solution') -> bool:
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return f"behaviour: {self._behaviour}, fitness: {self._fitness}, parameters: {self._parameters}, metadata: {self._metadata}"
    
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
        self._interpolator = None

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
    
    def get_bin_index(self, features: List[float]) -> Tuple[int, ...] | None:
        """
        returns a tuple with the indices of the bin corresponding to the features. If invalid features i.e. outside of the defined space return None
        """
        indices = []
        for res, bound, feature, feature_name in zip(self._resolutions, self._feature_bounds, features, self._feature_names):
            if feature < bound[0] or feature > bound[1]:
                print(f"Warning: Solution ignored because it falls outside of the cells, {feature_name}: {feature}")
                return None
            comparator = np.linspace(bound[0], bound[1], res+1)
            index = np.argmax(feature < comparator)-1
            indices.append(index)
        return tuple(indices)
    
    def get_middle_of_bin(self, bin_index: Tuple[int]) -> List[float]:
        """
        Returns the middle of the features of the specified bin.

        :param bin_index: The index of the bin.
        :return: The middle of the features of the bin.
        """
        middle_features = []
        for i, index in enumerate(bin_index):
            feature_bound = self._feature_bounds[i]
            resolution = self._resolutions[i]
            feature_range = feature_bound[1] - feature_bound[0]
            bin_size = feature_range / resolution
            middle_feature = feature_bound[0] + (index + 0.5) * bin_size
            middle_features.append(middle_feature)
        return middle_features
    
    def get_closest_solutions(self, 
                              feature: np.ndarray,
                              k: int,
                              ) -> List[Tuple[Solution, float, np.ndarray]]:
        """
        Returns the k closest solutions based on the feature space of the middle of the bin with bin_index.

        :param feature: The feature to compare to.
        :param k: The number of closest solutions to return.
        :return: A list of the k closest tuples in ascending order. tuple[Solution, distance, vector]
        """
        distances = []
        for sol in self:
            sol_features = sol.behaviour
            distance = np.linalg.norm(np.array(feature) - np.array(sol_features))
            vector = np.array(feature) - np.array(sol_features)
            distances.append((sol, distance, vector))
        distances.sort(key=lambda x: x[1])  # lowest first
        closest_solutions = [(sol, distance, vector) for sol, distance, vector in distances[:k]]
        return closest_solutions
    
    def interpolate(self, 
                    features: np.ndarray, 
                    k:int = 4,
                    ) -> np.ndarray:
        if not hasattr(self, "_interpolator") or self._interpolator == None:
            features_ = np.array([sol.behaviour for sol in self])
            parameters = np.array([sol.parameters for sol in self])
            self._interpolator = LinearNDInterpolator(points=features_, values=parameters)
            self._interpolator = RBFInterpolator(y=features_, d=parameters)

        interpolated_parameters = np.zeros_like(self._parameter_bounds[0])
        interpolated_parameters = self._interpolator(features)    # rows are the parameter samples, nan this means that the values are outside of the convex hull
        if np.isnan(interpolated_parameters).any():
            print("no suitable parameters are found by the interpolator, returning the closest one.")
            return self.get_closest_solutions(features, k=1)[0][0].parameters
        else:
            return interpolated_parameters[0]
    
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
        # other_solution = self.get_symmetric_solution(solution)
        # all_solutions.append(other_solution)

        # Add all solutions to the archive
        for sol in all_solutions:
            index = self.get_bin_index(sol.behaviour)
            if index == None: continue  # feature not in the defined space
            self._solutions[index].append(sol)
            if len(self._solutions[index]) > self._max_items_per_bin:
                self._solutions[index].remove(min(self._solutions[index]))
    
    def remove(self,
               index: Tuple[int, ...],
               ) -> None:
        """
        Remove the solution in the specified bin.
        """
        self._solutions[index].pop()

    
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
        df = pd.DataFrame({x_label: x_data, y_label: y_data, 'index_bin': index_bin_data, 'fitness': fitness_data, 'index': index_data})

        # Add scatter plot for data points
        px_fig = px.scatter(df, x=x_label, y=y_label, hover_data=df.columns, color='fitness')
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
    
    def plot_grid_3d(self,
                     x_label: str,
                     y_label: str,
                     z_label: str,
                     title: str = "MAP-Elites Archive",
                     store: str | None = None,
                     ) -> None:
        """
        Plot the MAP-Elites archive in 3D.

        :param x_label: The label for the x-axis.
        :param y_label: The label for the y-axis.
        :param z_label: The label for the z-axis.
        :param title: The title of the plot.
        :param store: The directory and filename to store the plot as an HTML file. If None, the plot is not stored.
        """
        assert x_label in self._feature_names, "The x_label must be one of the feature names."
        assert y_label in self._feature_names, "The y_label must be one of the feature names."
        assert z_label in self._feature_names, "The z_label must be one of the feature names."
        x_index = self._feature_names.index(x_label)
        y_index = self._feature_names.index(y_label)
        z_index = self._feature_names.index(z_label)
        x_data = []
        y_data = []
        z_data = []
        index_bin_data = []
        fitness_data = []
        index_data = []
        for sol in self:
            x_data.append(sol.behaviour[x_index])
            y_data.append(sol.behaviour[y_index])
            z_data.append(sol.behaviour[z_index])
            index_bin_data.append(self.get_bin_index(sol.behaviour))
            fitness_data.append(sol.fitness)
            index_data.append(self._solutions[self.get_bin_index(sol.behaviour)].index(sol))
        fig = go.Figure(data=go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                color=fitness_data,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Fitness')
            ),
            hovertemplate=
            '<b>'+x_label+': %{x}</b><br>' +
            '<b>'+y_label+': %{y}</b><br>' +
            '<b>'+z_label+': %{z}</b><br>' +
            '<b>index_bin: (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})</b><br>' +
            '<b>index: %{customdata[3]}</b><br>' +
            '<b>fitness: %{marker.color}</b><br>',
            customdata=np.column_stack((index_bin_data, index_data))
        ))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label
            ),
            font=dict(size=20),  # Increase font size
        )
        
        # Set visible range for 3D plot
        fig.update_scenes(xaxis_range=[self._feature_bounds[x_index][0], self._feature_bounds[x_index][1]])
        fig.update_scenes(yaxis_range=[self._feature_bounds[y_index][0], self._feature_bounds[y_index][1]])
        fig.update_scenes(zaxis_range=[self._feature_bounds[z_index][0], self._feature_bounds[z_index][1]])
        
        if store is not None:
            fig.write_html(store)
        fig.show()
    
    def plot_distance_neighbours_distribution(self,
                            parameter_name: str,
                            title: str = "MAP-Elites Archive",
                            filename: str | None = None,
                            show: bool = False,
                            ) -> None:
        """
        For the given parameter, plot the distribution of the differences between the parameter value and the values of the 4 neighbouring solutions. 
        parameter_name: str: The name of the parameter to plot the distribution of the differences.
        title: str: The title of the plot.
        filename: str: The directory and filename to store the plot as an HTML file. If None, the plot is not stored.
        show: bool: Whether to show the plot.
        """
        assert parameter_name in self._parameter_names, "The parameter_name must be one of the parameter names."
        parameter_index = self._parameter_names.index(parameter_name)
        differences = []
        for sol in self:
            index = self.get_bin_index(sol.behaviour)
            if index is None:
                continue
            neighbours = [(index[0] + 1, index[1], index[2]), 
                  (index[0] - 1, index[1], index[2]), 
                  (index[0], index[1] + 1, index[2]), 
                  (index[0], index[1] - 1, index[2]),
                  (index[0], index[1], index[2] + 1),
                  (index[0], index[1], index[2] - 1)]
            for neighbour in neighbours:
                if neighbour in self._solutions.keys():
                    for neighbour_sol in self._solutions[neighbour]:
                        difference = abs(neighbour_sol.parameters[parameter_index] - sol.parameters[parameter_index])
                        differences.append(difference)
        fig = go.Figure(data=[go.Histogram(x=differences)])
        fig.update_layout(
            title=title,
            xaxis_title=f"Difference in {parameter_name}",
            yaxis_title="Frequency",
            font=dict(
            size=25,
            )
        )
        if show:
            fig.show()
        if filename is not None:
            fig.write_html(f"{filename}.html")


    def get_bins(self) -> List[Tuple[int, ...]]:
        """
        Returns a list of all the bins in the archive ie. the tuples corresponding to the indices.
        """
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
                 archive_file: str = "archive.pkl",
                 ) -> None:
        """
        Initialize the MAP-Elites algorithm.

        :param archive: The archive to use.
        :param archive_file: The name of the file to store the archive.
        """
        self._archive = archive
        self._archive_file = archive_file
        self._info = {'number_of_individuals': [], 'avg_fitness':[]}
    
    @property
    def archive(self):
        return self._archive


    def ask(self) -> np.ndarray:
        """
        Generates a new parameter sample
        """
        np.random.seed()
        if len(self._archive) < 30:
            return np.array([bound[0] + rand * (bound[1]-bound[0]) for bound, rand in zip(self._archive.parameter_bounds, np.random.rand(len(self._archive.parameter_bounds)))])
        else:
            np.random.seed()    # otherwise the same random number is generated every time -> why?
            ## selection
            # Make probabilities features for choosing a bin
            bins = self._archive.get_bins()
            bins_num_sol = np.array([self._archive.num_sol_per_bin()[bin] for bin in bins])
            # normalize
            bins_num_sol = bins_num_sol/np.sum(bins_num_sol)
            probs = (1-bins_num_sol)
            probs = probs/np.sum(probs)
            # choose a bin for which we want to find the parameters to attain the behaviour of that bin
            bin_index = np.random.choice(range(len(bins)), p=probs)
            bin_index = bins[bin_index]
            bin: List[Solution] = self._archive.solutions[bin_index]
            # choose 2 solutions as parents
            if np.random.rand() < 0.2:   # 20% of the time, 2 random parents are chosen
                solutions = list(self._archive)
                random_keys = np.random.randint(0, len(self._archive), 2)
                parent1 = next(itertools.islice(solutions, random_keys[0], random_keys[0]+1)).parameters
                parent2 = next(itertools.islice(solutions, random_keys[1], random_keys[1]+1)).parameters
            else:
                sol: List[Tuple[Solution, float, np.ndarray]] = self._archive.get_closest_solutions(feature=self.archive.get_middle_of_bin(bin_index), k=2)
                parent1 = sol[0][0].parameters
                parent2 = sol[1][0].parameters

            ## crossover
            par_to_change = np.random.choice(range(len(parent1)), int(len(parent1)/2), [1/len(parent1) for _ in range(len(parent1))])
            child = parent1.copy()
            child[par_to_change] = parent2[par_to_change]

            ## mutation
            child += np.random.normal(0., 0.08, len(child))
            child = np.clip(child, 0, 1)
            return child


    def tell(self, solutions: List[Solution]) -> None:
        """
        Update the grid with a new solution.

        :param solutions: a list of solutions to add to the archive.
        """
        # for sol in solutions:
        #     assert isinstance(sol, Solution), \
        #         f"there is a list element that is not of type Solution, type: {type(sol)}"
        for sol in solutions:
            self._archive.add_solution(solution=sol)
            self._info['number_of_individuals'].append(len(self._archive))
            self._info['avg_fitness'].append(np.mean([sol.fitness for sol in self._archive]))
        
        self._archive.store(self._archive_file)
    
    def optimization_info(self, store: str | None = None):
        """
        plots the avg fitness and number of individuals of the history 

        :param store: The directory and filename to store the plot as an HTML file. If None, the plot is not stored.
        """
        import plotly.graph_objects as go

        number_of_individuals = self._info['number_of_individuals']
        avg_fitness = self._info['avg_fitness']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(number_of_individuals))),
            y=number_of_individuals,
            mode='lines',
            name='Number of Individuals',
            yaxis='y1'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(len(avg_fitness))),
            y=avg_fitness,
            mode='lines',
            name='Average Fitness',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Archive through time',
            xaxis_title='Number of total Individuals',
            yaxis=dict(
            title='Number of Individuals in Archive',
            side='left',
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14)  # Increase font size
            ),
            yaxis2=dict(
            title='Fitness',
            side='right',
            overlaying='y',
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=20),  # Increase font size
            ),
        )

        if store is not None:
            fig.write_html(store)
        fig.show()


if __name__ == "__main__":
    archive = Archive(feature_bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)], resolutions=[2, 2, 1], feature_names=["pitch", "yawn", "roll"])
    me = MapElites(archive)
    archive.plot_grid(x_label="pitch", y_label="roll")