from modularcoevolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator, AgentType
from modularcoevolution.utilities.specialtypes import GenotypeID

from typing import Any, Generic, Type

import math
import random

# if TYPE_CHECKING:
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.selfadaptivewrapper import SelfAdaptiveWrapper
from modularcoevolution.utilities.datacollector import DataCollector


# TODO: Rename variables
# Normal genetic programming evolutionary algorithm
class EvolutionGenerator(BaseEvolutionaryGenerator, Generic[AgentType]):
    def __init__(self, agent_class: Type[AgentType],
                 population_name: str,
                 initial_size: int,
                 children_size: int,
                 agent_parameters: dict[str, Any] = None,
                 genotype_parameters: dict[str, Any] = None,
                 mutation_fraction: float = 0.25,
                 recombination_fraction: float = 0.75,
                 diversity_weight: float = 0,
                 diverse_elites: bool = False,
                 compute_diversity: bool = False,
                 seed: Any = None,
                 data_collector: DataCollector = None,
                 copy_survivor_objectives: bool = False,
                 reevaluate_per_generation: bool = True,
                 using_hall_of_fame: bool = False,
                 tournament_size: int = 2,
                 past_population_width: int = 1):
        super().__init__(agent_class, population_name, initial_size, agent_parameters=agent_parameters,
                         genotype_parameters=genotype_parameters, seed=seed,
                         data_collector=data_collector, copy_survivor_objectives=copy_survivor_objectives,
                         reevaluate_per_generation=reevaluate_per_generation, using_hall_of_fame=using_hall_of_fame,
                         compute_diversity=compute_diversity, past_population_width=past_population_width)
        self.children_size = children_size
        self.mutation_fraction = mutation_fraction
        self.recombination_fraction = recombination_fraction
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
        random.shuffle(self.population)  # Python's list.sort maintains existing order between same-valued individuals, which can lead to stagnation in extreme cases such as all zero fitnesses

        if self.diversity_weight > 0:
            for genotype in self.population:
                novelty = genotype.metrics["novelty"]
                if novelty > self.max_novelty:
                    self.max_novelty = novelty
        else:
            self.max_novelty = 0

        if self.diverse_elites:
            best = max(self.population, key=lambda x: x.fitness)
            self.population.sort(key=lambda x: self.calculate_diversity_fitness(x), reverse=True)
            self.population.remove(best)
            self.population.insert(0,
                                   best)  # Even with diverse elites, keep the absolute best individual as an elite no matter what.
        else:
            self.population.sort(key=lambda x: x.fitness, reverse=True)  # High fitness is good

        self.log_generation()

    def next_generation(self):
        next_generation = list()
        next_generation_set = set()
        for i in range(self.initial_size):
            if self.copy_survivor_objectives:
                survivor = self.population[i]
            else:
                survivor = self.population[i].clone()
            next_generation.append(survivor)
            next_generation_set.add(hash(survivor))

        num_mutation = int(math.ceil(self.mutation_fraction * self.children_size))
        num_recombination = int(math.floor(self.recombination_fraction * self.children_size))

        for i in range(self.children_size):
            unique = False
            child = None
            while not unique:
                if i < num_mutation or (isinstance(child, SelfAdaptiveWrapper) and random.random() <
                                        child.self_adaptive_parameters["mutation rate"]):
                    child = self.generate_mutation()
                else:
                    child = self.generate_recombination()
                if hash(child) not in next_generation_set:
                    unique = True
            next_generation.append(child)
            next_generation_set.add(hash(child))

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.past_populations.append(self.population[:self.past_population_width])
        self.population = next_generation
        self.population_size = self.initial_size + self.children_size
        for genotype in self.population:
            self.genotypes_by_id[genotype.id] = genotype
        if self.using_hall_of_fame:
            self.hall_of_fame.extend([self.population[i] for i in self.get_representatives_from_generation(self.generation, 1)])

        self.generation += 1

    def calculate_diversity_fitness(self, individual):
        if self.max_novelty == 0:
            return individual.fitness
        return individual.fitness * (1 + (self.diversity_weight * individual.metrics["novelty"] / self.max_novelty))

    def fitness_proportionate_selection(self):
        best_fitness = self.calculate_diversity_fitness(self.population[0])
        worst_fitness = self.calculate_diversity_fitness(self.population[-1])
        if best_fitness == worst_fitness:
            return random.choice(self.population)
        while True:
            choice = random.choice(self.population)
            if random.random() < (self.calculate_diversity_fitness(choice) - worst_fitness) / (
                    best_fitness - worst_fitness):
                return choice

    def tournament_selection(self, k):
        tournament = random.sample(self.population, k)
        tournament.sort(key=self.calculate_diversity_fitness, reverse=True)
        return tournament[0]

    def generate_mutation(self):
        #parent = self.fitness_proportionate_selection()
        parent = self.tournament_selection(self.tournament_size)
        child = parent.clone()
        child.mutate()
        return child

    def generate_recombination(self):
        #parent = self.fitness_proportionate_selection()
        #donor = self.fitness_proportionate_selection()
        parent = self.tournament_selection(self.tournament_size)
        donor = self.tournament_selection(self.tournament_size)
        child = parent.clone()
        child.recombine(donor)
        return child

    @staticmethod
    def sorted_population(population: list[BaseGenotype]):
        return sorted(population, key=lambda x: x.fitness, reverse=True)
