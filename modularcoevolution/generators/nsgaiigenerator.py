#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import statistics
from typing import Type, Any

from modularcoevolution.generators.basegenerator import AgentType
from modularcoevolution.generators.evolutiongenerator import EvolutionGenerator
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.baseobjectivetracker import MetricConfiguration

import functools
import math
import random


# Note: these functions assume objective maximization
class NSGAIIGenerator(EvolutionGenerator[AgentType]):
    """A generator that uses the NSGA-II algorithm to evolve a population of agents with multiple objectives."""

    parsimony_objective: bool
    """Whether to automatically register and calculate a parsimony pressure objective
    (based on :meth:`BaseObjectiveTracker.get_fitness_modifier`)."""

    nondominated_fronts: list[list[BaseGenotype]]
    """A list of non-dominated fronts for the current generation as lists of genotypes,
    starting with the pareto front."""

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            initial_size: int,
            parsimony_objective: bool = False,
            **kwargs
    ):
        super().__init__(
            agent_class=agent_class,
            population_name=population_name,
            initial_size=initial_size,
            **kwargs
        )

        self.parsimony_objective = parsimony_objective
        if self.parsimony_objective:
            self._register_parsimony_objective()

        self.nondominated_fronts = list()

        self._register_front_metric()
        self._register_crowding_metric()

        if self.hall_of_fame_size > 0:
            raise NotImplementedError

    def end_generation(self) -> None:
        if self.parsimony_objective:
            for individual in self.population:
                # Use a fake 'raw fitness' value of 1, since some genotypes scale parsimony pressure by fitness.
                self.submit_metric(individual.id, "parsimony", individual.get_fitness_modifier(1))

        self.nondominated_fronts = nondominated_sort(self.population)

        crowding_distances = dict()
        for index, front in enumerate(self.nondominated_fronts):
            crowding_distances.update(calculate_crowding_distances(front))
            for individual in front:
                self.submit_metric(individual.id, "front", index)
        for individual in crowding_distances:
            self.submit_metric(individual.id, "crowding", crowding_distances[individual])

        super().end_generation()

    def log_generation(self):
        super().log_generation()
        pareto_front_objectives = []
        for individual in self.nondominated_fronts[0]:
            pareto_front_objectives.append(tuple(individual.objectives[objective] for objective in self.population[0].objectives))
        print(f"Pareto front: {sorted(pareto_front_objectives)}")


    def get_population_metrics(self) -> dict[str, Any]:
        population_metrics = super().get_population_metrics()
        population_metrics["front_sizes"] = [len(front) for front in self.nondominated_fronts]
        return population_metrics

    def get_metric_statistics(
            self,
            metric: str,
            configuration: MetricConfiguration,
            population_metrics: list
    ) -> dict[str, Any]:
        metric_statistics = super().get_metric_statistics(metric, configuration, population_metrics)
        pareto_metrics = [individual.metrics[metric] for individual in self.nondominated_fronts[0]]
        metric_statistics["pareto_median"] = statistics.median(pareto_metrics)
        metric_statistics["pareto_minimum"] = min(pareto_metrics)
        # Pareto maximum is identical to "maximum" in the superclass
        return metric_statistics

    @classmethod
    def sorted_population(cls, population: list[BaseGenotype]):
        return sorted(population, key=functools.cmp_to_key(crowded_comparison), reverse=True)

    def _register_front_metric(self) -> None:
        """Automatically register a nondominating front metric called ``"front"``."""
        metric_configuration: MetricConfiguration = {
            "name": "front",
            "is_objective": False,
            "repeat_mode": "replace",
            "log_history": False,
            "automatic": False,
            "add_fitness_modifier": False,
        }
        self.register_metric(metric_configuration, None)

    def _register_crowding_metric(self) -> None:
        """Automatically register a crowding distance metric called ``"crowding"``."""
        metric_configuration: MetricConfiguration = {
            "name": "crowding",
            "is_objective": False,
            "repeat_mode": "replace",
            "log_history": False,
            "automatic": False,
            "add_fitness_modifier": False,
        }
        self.register_metric(metric_configuration, None)

    def _register_parsimony_objective(self) -> None:
        """Automatically register a parsimony pressure objective called ``"parsimony"``."""
        metric_configuration: MetricConfiguration = {
            "name": "parsimony",
            "is_objective": True,
            "repeat_mode": "replace",
            "log_history": False,
            "automatic": False,
            "add_fitness_modifier": False,
        }
        self.register_metric(metric_configuration, None)


def nondominated_sort(population):
    dominated_sets = {individual: [] for individual in population}
    domination_counters = {individual: 0 for individual in population}
    nondominated_fronts = [[]]
    for individual in population:
        for other in population:
            domination = domination_comparison(individual, other)
            if domination > 0:
                dominated_sets[individual].append(other)
            elif domination < 0:
                domination_counters[individual] += 1
        if domination_counters[individual] == 0:
            nondominated_fronts[0].append(individual)

    front = 0
    while len(nondominated_fronts[front]) > 0:
        nondominated_fronts.append([])
        for dominator in nondominated_fronts[front]:
            for dominated in dominated_sets[dominator]:
                domination_counters[dominated] -= 1
                if domination_counters[dominated] == 0:
                    nondominated_fronts[front + 1].append(dominated)
        front += 1
    nondominated_fronts.pop(-1)
    return nondominated_fronts


def calculate_crowding_distances(front_population):
    crowding_distances = {individual: 0.0 for individual in front_population}
    for objective in front_population[0].objectives:
        objective_min = min(individual.objectives[objective] for individual in front_population)
        min_individual = [individual for individual in front_population if individual.objectives[objective] == objective_min][0]
        objective_max = max(individual.objectives[objective] for individual in front_population)
        max_individual = [individual for individual in front_population if individual.objectives[objective] == objective_max][0]

        def sorting_comparison(individual_1, individual_2):
            objective_1 = individual_1.objectives[objective]
            objective_2 = individual_2.objectives[objective]
            objective_difference = objective_1 - objective_2
            if objective_difference != 0:
                return objective_difference
            elif objective_1 == objective_max:
                # Put the newest ID last (at the extreme)
                # Slightly encourages genetic drift?
                return individual_1.id - individual_2.id
            else:
                # Put the newest ID first (at the extreme)
                return individual_2.id - individual_1.id
        sorted_population = list(front_population)
        sorted_population.sort(key=functools.cmp_to_key(sorting_comparison))
        crowding_distances[sorted_population[0]] = math.inf
        objective_min = sorted_population[0].objectives[objective]
        crowding_distances[sorted_population[-1]] = math.inf
        objective_max = sorted_population[-1].objectives[objective]

        for i in range(1, len(sorted_population) - 1):
            individual = sorted_population[i]
            if objective_max == objective_min:
                crowding_distances[individual] = math.inf  # Same as the boundaries?
            else:
                crowding_distances[individual] += (sorted_population[i+1].objectives[objective] - sorted_population[i-1].objectives[objective]) / (objective_max - objective_min)
    return crowding_distances


def domination_comparison(individual_1: BaseGenotype, individual_2: BaseGenotype) -> int:
    comparisons = [individual_1.objectives[objective] - individual_2.objectives[objective] for objective in individual_1.objectives]
    if all(comparison >= 0 for comparison in comparisons):
        if any(comparison > 0 for comparison in comparisons):
            #  No objective of individual_1 is worse, and at least one is better
            return 1
        else:
            return 0
    elif all(comparison <= 0 for comparison in comparisons):
        if any(comparison < 0 for comparison in comparisons):
            #  No objective of individual_2 is worse, and at least one is better
            return -1
        else:
            return 0
    else:
        return 0



def crowded_comparison(individual_1: BaseGenotype, individual_2: BaseGenotype) -> int:
    domination = domination_comparison(individual_1, individual_2)
    if domination != 0:
        return domination
    else:
        comparison = individual_1.metrics['crowding'] - individual_2.metrics['crowding']
        if comparison > 0:
            return 1
        elif comparison < 0:
            return -1
        else:
            return 0
