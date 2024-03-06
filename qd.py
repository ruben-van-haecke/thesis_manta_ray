from qdpy import algorithms, containers, benchmarks, plots
from qdpy.algorithms.base import QDAlgorithm, Container
from qdpy.phenotype import Individual, IndividualLike, Fitness
from typing import Optional, Generator, Union, Sequence
import numpy as np
from typing import List

class QDWrapper():
    def __init__(self, algorithm: QDAlgorithm, *args, **kwargs,):
        self._alg: QDAlgorithm = algorithm(**kwargs)
        self._current_individuals: List[IndividualLike] = []
        self._population_size = kwargs["batch_size"]

    @property
    def grid(self):
        return self._alg.container.size_str()
    
    def ask(self) -> np.ndarray:
        print("ask in qdWrapper")
        self._individual: IndividualLike = self._alg.ask()
        self._current_individuals.append(self._individual)
        return np.array(self._individual)
    
    def tell(self, evaluations: np.ndarray) -> None:
        print("tell in qdWrapper")
        for i in range(self._population_size):
            fitness = Fitness(values=evaluations[i])
            self._current_individuals[i].fitness = fitness
            self._alg.tell(self._current_individuals[i])

        self._current_individuals = []
        return None
    
    def results(self):
        return self._alg.tell_container_entries()


if __name__ == "__main__":
    from qdpy import algorithms, containers, benchmarks, plots

    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
    grid = containers.Grid(shape=(64,64), max_items_per_bin=1, fitness_domain=((0., 1.),), features_domain=((0., 1.), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=60000, batch_size=500,
            dimension=3, optimisation_task="maximisation")

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.AlgorithmLogger(algo)

    # Define evaluation function
    eval_fn = algorithms.partial(benchmarks.illumination_rastrigin_normalised,
            nb_features = len(grid.shape))

    # Run illumination process !
    best = algo.optimise(eval_fn)

    # Print results info
    print(algo.summary())

    # Plot the results
    plots.default_plots_grid(logger)

    print("All results are available in the '%s' pickle file." % logger.final_filename)
