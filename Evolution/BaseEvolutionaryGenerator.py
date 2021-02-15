import ast

from Evolution.BaseGenerator import *
from Evolution.BaseEvolutionaryAgent import *


# Generic evolutionary algorithm
class BaseEvolutionaryGenerator(BaseGenerator):
    def __init__(self, agent_class, initial_size, agent_parameters=None, genotype_parameters=None, seed=None, fitness_function=None, data_collector=None, copy_survivor_objectives=False, reevaluate_per_generation=True, using_hall_of_fame=True):
        super().__init__()
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        if self.agent_parameters is None:
            self.agent_parameters = dict()
        self.genotype_parameters = genotype_parameters
        if self.genotype_parameters is None:
            self.genotype_parameters = dict()
        self.initial_size = initial_size
        self.fitness_function = fitness_function
        self.max_novelty = 0
        self.seed = seed
        self.data_collector = data_collector
        self.copy_survivor_objectives = copy_survivor_objectives
        self.reevaluate_per_generation = reevaluate_per_generation
        assert issubclass(agent_class, BaseEvolutionaryAgent)
        self.genotype_class = agent_class.genotypeClass()

        self.generation = 0
        self.population_size = self.initial_size
        self.population = list()
        self.past_populations = list()
        self.using_hall_of_fame = using_hall_of_fame
        self.hall_of_fame = list()

        population_set = set()
        for i in range(self.initial_size):
            default_parameters = self.agent_class.genotypeDefaultParameters()
            default_parameters.update(self.genotype_parameters)
            if self.seed is not None and i < len(self.seed):
                parameters = default_parameters.copy()
                parameters.update(self.seed[i])
                individual = self.genotype_class(parameters)
                self.population.append(individual)
                population_set.add(hash(individual))
            else:
                unique = False
                individual = None
                while not unique:
                    individual = self.genotype_class(default_parameters.copy())
                    if hash(individual) not in population_set:
                        unique = True
                self.population.append(individual)
                population_set.add(hash(individual))
        for individual in self.population:
            self.claim_ID(individual)

        self.evaluation_lists = dict()

    def __getitem__(self, index):
        agent = self.agent_class(genotype=self.population[index], active=False, **self.agent_parameters)
        return agent

    def get_from_generation(self, generation, index):
        if generation == len(self.past_populations):
            agent = self.agent_class(genotype=self.population[index], active=False, **self.agent_parameters)
        elif generation == -1:
            agent = self.agent_class(genotype=self.hall_of_fame[index], active=False, **self.agent_parameters)
        else:
            agent = self.agent_class(genotype=self.past_populations[generation][index], active=False, **self.agent_parameters)
        return agent
    
    def get_individuals_to_test(self):
        return [(self.generation, i) for i in range(self.population_size) if self.reevaluate_per_generation or not self.population[i].evaluated] + \
               [(-1, i) for i in range(len(self.hall_of_fame))]

    def set_objectives(self, index, objectives, average_flags=None, average_fitness=False, opponent=None,
                       evaluation_number=None, inactive_objectives=None):
        if average_flags is None:
            average_flags = {objective: False for objective in objectives}
        if opponent is not None:
            assert isinstance(opponent, BaseEvolutionaryAgent)
        individual = self.population[index]
        individual.set_objectives(objectives, average_flags, inactive_objectives)
        if len(individual.get_active_objectives()) > 0:
            if self.fitness_function is not None:
                fitness_modifier = individual.get_fitness_modifier()
                fitness = self.fitness_function(individual.get_active_objectives()) + fitness_modifier
                individual.set_fitness(fitness, average_fitness)
                individual.metrics["quality"] = individual.fitness
            else:
                fitness_modifier = individual.get_fitness_modifier()
                fitness = sum(individual.get_active_objectives().values()) / len(
                    individual.get_active_objectives()) + fitness_modifier
                individual.set_fitness(fitness, average_fitness)
                individual.metrics["quality"] = individual.fitness
        if "novelty" not in individual.metrics:
            individual.metrics["novelty"] = self.get_diversity(index)
        if individual.ID not in self.evaluation_lists:
            self.evaluation_lists[individual.ID] = list()
        self.evaluation_lists[individual.ID].append((objectives, evaluation_number))

        if self.data_collector is not None:
            agent_type_name = self.agent_class.agent_type_name
            evaluations = [evaluation_ID for fitness, evaluation_ID in
                           self.evaluation_lists[individual.ID]]

            self.data_collector.set_individual_data(agent_type_name, individual.ID, individual.get_raw_genotype(),
                                                    evaluations, individual.objective_statistics, individual.metrics,
                                                    [parent.ID for parent in individual.parents],
                                                    individual.creation_method)

    def generate_individual(self, parameter_string):
        raise NotImplementedError("generate_individual is deprecated and will be removed from the base class unless it turns out something needs it.")

    def get_diversity(self, referenceID=None, samples=10, presorted=False):
        if referenceID is not None:
            reference = self.population[referenceID]
        elif presorted:
            reference = self.population[0]  # Avoid redoing search for best if best is known
        else:
            reference = None
        return reference.diversity_function(self.population, reference, samples)
