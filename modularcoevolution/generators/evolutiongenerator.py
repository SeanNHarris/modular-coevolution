import warnings

from modularcoevolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator, AgentType
from modularcoevolution.utilities.specialtypes import GenotypeID

from typing import Any, Generic, Type

import math
import random

# if TYPE_CHECKING:
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.selfadaptivewrapper import SelfAdaptiveWrapper
from modularcoevolution.utilities.datacollector import DataCollector


# Normal genetic programming evolutionary algorithm
class EvolutionGenerator(BaseEvolutionaryGenerator, Generic[AgentType]):
    """A generator that uses a standard evolutionary algorithm to evolve agents."""

    children_size: int
    """The number of children to create each generation (lambda)."""
    mutation_fraction: float
    """The fraction of the children that will be mutated (the mutation rate)."""
    recombination_fraction: float
    """The fraction of the children that will be created through recombination.
    All others will be created from a single parent."""
    mutate_after_recombine: bool
    """If true, mutation can be applied to children created through recombination, as in standard evolution.
    If false, mutation and recombination will be mutually exclusive, which is standard for genetic programming."""
    diversity_weight: float
    """If non-zero, the fitness of each individual will be multiplied by `(1 + diversity_weight * novelty / max_novelty)`."""
    diverse_elites: bool
    """If true, survival selection will be based on diversity fitness rather than raw fitness.
    The absolute best individual will always be sorted to the front of the population, regardless of diversity fitness,
    in order to prevent the loss of the best solution."""
    tournament_size: int
    """The number of individuals compared during tournament selection (k)."""

    max_novelty: float
    """The maximum novelty value observed in the population. Used to normalize novelty values for diversity fitness."""

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            initial_size: int,
            children_size: int,
            mutation_fraction: float = 0.25,
            mutate_after_recombine: bool = True,
            diversity_weight: float = 0,
            diverse_elites: bool = False,
            tournament_size: int = 2,
            **kwargs
    ):
        super().__init__(agent_class, population_name, initial_size, **kwargs)
        self.children_size = children_size
        self.mutation_fraction = mutation_fraction
        self.mutate_after_recombine = mutate_after_recombine

        if mutate_after_recombine:
            self.recombination_fraction = 1.0
        else:
            self.recombination_fraction = 1.0 - mutation_fraction

        self.diversity_weight = diversity_weight
        self.diverse_elites = diverse_elites
        self.tournament_size = tournament_size

        self.max_novelty = 0

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False)\
            -> list[GenotypeID]:
        if generation == len(self.past_populations):
            sorted_population = self.population.copy()
        else:
            sorted_population = self.sorted_population(self.past_populations[generation])
        if force:
            indices = [i % len(sorted_population) for i in range(amount)]
        else:
            indices = range(min(amount, len(sorted_population)))
        return [sorted_population[i].id for i in indices]

    def end_generation(self) -> None:
        super().end_generation()
        random.shuffle(self.population)  # Python's list.sort maintains existing order between same-valued individuals, which can lead to stagnation in extreme cases such as all zero fitnesses

        self.population = self.sorted_population(self.population)
        self.past_populations.append(self.population[:self.past_population_width])
        self.log_generation()

        # Clear out genotypes set not to be stored after logging
        for genotype in self.population[self.past_population_width:]:
            if genotype not in self.hall_of_fame:
                del self.genotypes_by_id[genotype.id]


    def next_generation(self):
        super().next_generation()
        # Population was already sorted in end_generation
        # Re-sort using diversity fitness
        if self.diverse_elites:
            best = max(self.population, key=lambda x: x.fitness)
            self.population.sort(key=lambda x: self.calculate_diversity_fitness(x), reverse=True)
            self.population.remove(best)
            self.population.insert(0, best)
            # Even with diverse elites, keep the absolute best individual as an elite no matter what.

        if self.diversity_weight > 0:
            for genotype in self.population:
                novelty = genotype.metrics["novelty"]
                if novelty > self.max_novelty:
                    self.max_novelty = novelty
        else:
            self.max_novelty = 0

        next_generation = list()
        next_generation_set = set()
        for i in range(self.initial_size):
            if self.copy_survivor_objectives:
                survivor = self.population[i]
            else:
                survivor = self.population[i].clone()
            next_generation.append(survivor)
            next_generation_set.add(hash(survivor))

        num_recombination = int(math.floor(self.recombination_fraction * self.children_size))
        num_mutation = int(math.ceil(self.mutation_fraction * self.children_size))

        for i in range(self.children_size):
            recombine = i < num_recombination
            if self.mutate_after_recombine:
                mutate = random.random() < self.mutation_fraction
            else:
                mutate = num_recombination <= i < num_recombination + num_mutation
            if not recombine and not mutate:
                raise RuntimeError("No mutation or recombination occurred. This should not happen.")

            # Ensure that the child is unique, to prevent duplicates in the population
            unique = False
            child = None
            failure_count = 0
            while not unique:
                if recombine:
                    child = self.generate_recombination()
                else:
                    child = self.generate_clone()

                # Special case for self-adaptive mutation rate
                if isinstance(self.population[0], SelfAdaptiveWrapper):
                    mutate = random.random() < child.self_adaptive_parameters["mutation rate"]

                if mutate:
                    child.mutate()
                if hash(child) not in next_generation_set:
                    unique = True
                else:
                    failure_count += 1
                    if failure_count > self.children_size:
                        warnings.warn("Repeatedly unable to generate a unique child. Relaxing constraint.")
                        unique = True  # A lie
            next_generation.append(child)
            next_generation_set.add(hash(child))

        self.population = next_generation
        for genotype in self.population:
            self.genotypes_by_id[genotype.id] = genotype

        if self.using_hall_of_fame:
            self.hall_of_fame.extend([self.population[i] for i in self.get_representatives_from_generation(self.generation, 1)])

        self.generation += 1

    def calculate_diversity_fitness(self, individual: BaseGenotype):
        """Calculate a value from fitness weighted by novelty, for use in diversity-based selection.

        Args:
            individual: The individual to calculate diversity fitness for.

        Returns:
            The original fitness multiplied by `(1 + diversity_weight * novelty / max_novelty)`.
        """
        if self.max_novelty == 0:
            return individual.fitness
        return individual.fitness * (1 + (self.diversity_weight * individual.metrics["novelty"] / self.max_novelty))

    def fitness_proportionate_selection(self) -> BaseGenotype:
        """Select an individual from the population using fitness-proportionate selection.
        * Todo: Rewrite and check this function.

        Returns:
            The selected individual.
        """
        best_fitness = self.calculate_diversity_fitness(self.population[0])
        worst_fitness = self.calculate_diversity_fitness(self.population[-1])
        if best_fitness == worst_fitness:
            return random.choice(self.population)
        while True:
            choice = random.choice(self.population)
            if random.random() < (self.calculate_diversity_fitness(choice) - worst_fitness) / (
                    best_fitness - worst_fitness):
                return choice

    def tournament_selection(self, k: int) -> BaseGenotype:
        """Select an individual from the population using k-tournament selection.
        `k` individuals are selected at random from the population, and the one with the highest fitness is returned.
        Results in a very gentle selection pressure with low `k`.

        Args:
            k: The number of individuals to compare in each tournament.

        Returns:
            The selected individual.
        """
        tournament = random.sample(self.population, k)
        if self.diversity_weight > 0:
            tournament.sort(key=self.calculate_diversity_fitness, reverse=True)
        else:
            tournament = self.sorted_population(tournament)
        return tournament[0]

    def generate_clone(self) -> BaseGenotype:
        """Generate a new individual by cloning a selected parent.

        Returns:
            A new child individual produced through :meth:`BaseGenotype.clone`.
        """
        parent = self.tournament_selection(self.tournament_size)
        child = parent.clone()
        return child

    def generate_recombination(self) -> BaseGenotype:
        """Generate a new individual by recombining two selected parents.

        Returns:
            A new child individual produced through :meth:`BaseGenotype.recombine`.
        """
        parent = self.tournament_selection(self.tournament_size)
        donor = self.tournament_selection(self.tournament_size)
        child = parent.clone()
        child.recombine(donor)
        return child

    @classmethod
    def sorted_population(cls, population: list[BaseGenotype]) -> list[BaseGenotype]:
        """Defines the sorting order for individuals.
        By default, this sorts by raw fitness.

        Args:
            population: The individuals to sort.

        Returns:
            A sorted copy of the input.
        """
        return sorted(population, key=lambda x: x.fitness, reverse=True)
