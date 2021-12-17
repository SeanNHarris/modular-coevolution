from Evolution.BaseEvolutionaryAgent import BaseEvolutionaryAgent
from Evolution.BaseEvolutionaryGenerator import BaseEvolutionaryGenerator
from alternate_genotypes.SelfAdaptiveWrapper import SelfAdaptiveWrapper

import math
import random


# TODO: Rename variables
# Normal genetic programming evolutionary algorithm
class EvolutionGenerator(BaseEvolutionaryGenerator):
    def __init__(self, agent_class, initial_size, children_size, agent_parameters=None, genotype_parameters=None, mutation_fraction=0.25,
                 recombination_fraction=0.75,
                 parsimony_weight=0, diversity_weight=0, diverse_elites=False, seed=None, fitness_function=None,
                 data_collector=None, copy_survivor_objectives=False, reevaluate_per_generation=True, using_hall_of_fame=True):
        super().__init__(agent_class, initial_size, agent_parameters=agent_parameters, genotype_parameters=genotype_parameters, seed=seed, fitness_function=fitness_function,
                         data_collector=data_collector, copy_survivor_objectives=copy_survivor_objectives, reevaluate_per_generation=reevaluate_per_generation, using_hall_of_fame=using_hall_of_fame)
        self.children_size = children_size
        self.mutation_fraction = mutation_fraction
        self.recombination_fraction = recombination_fraction
        self.parsimony_weight = parsimony_weight
        self.diversity_weight = diversity_weight
        self.diverse_elites = diverse_elites
        self.max_novelty = 0

    def get_fitness(self, index):
        return self.population[index].fitness

    # Returns indices
    def get_representatives_from_generation(self, generation, amount, force=False):
        if force:
            return [i % len(self.past_populations[generation]) for i in range(amount)]
        else:
            return range(min(amount, len(self.past_populations[generation])))

    # Creates a new generation
    def next_generation(self, result_log=None, agent_log=None):
        random.shuffle(self.population)  # Python's list.sort maintains existing order between same-valued individuals, which can lead to stagnation in extreme cases such as all zero fitnesses

        for i in range(len(self.population)):
            novelty = self.get_diversity(i)
            if novelty > self.max_novelty:
                self.max_novelty = novelty
            self.population[i].metrics["novelty"] = novelty

        if self.diverse_elites:
            best = max(self.population, key=lambda x: x.fitness)
            self.population.sort(key=lambda x: self.calculate_diversity_fitness(x), reverse=True)
            self.population.remove(best)
            self.population.insert(0,
                                   best)  # Even with diverse elites, keep the absolute best individual as an elite no matter what.
        else:
            self.population.sort(key=lambda x: x.fitness, reverse=True)  # High fitness is good

        best_fitness = max(self.population, key=lambda individual: individual.fitness).fitness
        average_fitness = sum([individual.fitness for individual in self.population]) / len(self.population)
        diversity = self.population[0].metrics["novelty"]

        if result_log is not None:
            result_columns = ["Generation", "Best Fitness", "Average Fitness", "Diversity"]
            result_log_format = "".join(
                ["{{!s:<{0}.{1}}}".format(len(column) + 4, len(column)) for column in result_columns]) + "\n"
            if self.generation == 0:
                result_log.truncate(0)
                result_log.write(result_log_format.format(*result_columns))

            result_log.write(result_log_format.format(self.generation, best_fitness, average_fitness, diversity))
            result_log.flush()

        if agent_log is not None:
            agent_columns = ["ID", "Fitness", "Novelty", "Parent IDs", "Evaluations, opponents, and fitnesses",
                             "Genotype"]
            agent_log_format = "".join(["{{!s:<{0}}}".format(len(column) + 4) for column in agent_columns]) + "\n"
            if self.generation == 0:
                agent_log.truncate(0)
            agent_log.write("---Generation {}---\n".format(self.generation))
            agent_log.write(agent_log_format.format(*agent_columns))

            for individual in self.population:
                parent_string = "({})".format(individual.creation_method)
                if len(individual.parents) == 1:
                    parent_string = "{} ({})".format(individual.parents[0].ID, individual.creation_method)
                elif len(individual.parents) == 2:
                    parent_string = "{}, {} ({})".format(individual.parents[0].ID, individual.parents[0].ID,
                                                         individual.creation_method)

                evaluation_list = self.evaluation_lists[individual.ID]
                evaluation_string = "["
                for i in range(len(evaluation_list)):
                    if i > 0:
                        evaluation_string += ", "
                    evaluation_string += str(evaluation_list[i][0])
                    if evaluation_list[i][1] is not None:
                        evaluation_string += " vs. " + str(evaluation_list[i][1])
                    if evaluation_list[i][2] is not None:
                        evaluation_string += " eval. " + str(evaluation_list[i][2])
                evaluation_string += "]"

                agent_log.write(
                    agent_log_format.format(individual.ID, individual.fitness, individual.metrics["novelty"],
                                            parent_string, evaluation_string, str(individual)))
            agent_log.flush()

        if self.data_collector is not None:
            if "agent_type_name" in self.agent_parameters:
                agent_type_name = self.agent_parameters["agent_type_name"]
            else:
                # Temporary while deprecating the class version of this variable
                agent_type_name = self.agent_class.agent_type_name
            population_IDs = [individual.ID for individual in self.population]
            objectives = dict()
            for objective in self.population[0].objectives:
                objective_sum = 0
                objective_sum_of_squares = 0
                objective_minimum = None
                objective_maximum = None
                for individual in self.population:
                    objective_value = individual.objectives[objective]
                    objective_sum += objective_value
                    objective_sum_of_squares += objective_value ** 2
                    if objective_minimum is None or objective_value < objective_minimum:
                        objective_minimum = objective_value
                    if objective_maximum is None or objective_value > objective_maximum:
                        objective_maximum = objective_value
                objective_mean = objective_sum / len(self.population)
                objective_standard_deviation = math.sqrt(
                    (objective_sum_of_squares - (len(self.population) * objective_mean ** 2)) / len(self.population))
                objectives[objective] = {"mean": objective_mean, "standard deviation": objective_standard_deviation,
                                         "minimum": objective_minimum, "maximum": objective_maximum}

            metrics = dict()
            for metric in self.population[0].metrics:
                if isinstance(metric, (int, float)):
                    metric_sum = 0
                    metric_sum_of_squares = 0
                    metric_minimum = None
                    metric_maximum = None
                    for individual in self.population:
                        metric_value = individual.metrics[metric]
                        metric_sum += metric_value
                        metric_sum_of_squares += metric_value ** 2
                        if metric_minimum is None or metric_value < metric_minimum:
                            metric_minimum = metric_value
                        if metric_maximum is None or metric_value > metric_maximum:
                            metric_maximum = metric_value
                    metric_mean = metric_sum / len(self.population)
                    metric_standard_deviation = math.sqrt(
                        (metric_sum_of_squares - (len(self.population) * metric_mean ** 2)) / len(self.population))
                    metrics[metric] = {"mean": metric_mean, "standard deviation": metric_standard_deviation,
                                       "minimum": metric_minimum, "maximum": metric_maximum}

            metrics.update({"diversity": diversity})
            self.data_collector.set_generation_data(agent_type_name, self.generation, population_IDs, objectives,
                                                    metrics)

        print("Best individual of this generation: (fitness score of " + str(self.population[0].fitness) + ")")
        print(self.population[0])

        print(str([individual.fitness for individual in self.population]))

        next_generation = list()
        next_generation_set = set()
        for i in range(self.initial_size):
            survivor = self.population[i].clone(copy_objectives=self.copy_survivor_objectives)
            next_generation.append(survivor)
            next_generation_set.add(survivor)

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
        self.past_populations.append(self.population)
        self.population = next_generation
        self.population_size = self.initial_size + self.children_size
        for individual in self.population:
            self.claim_ID(individual)
        if self.using_hall_of_fame:
            self.hall_of_fame.extend([self.population[i] for i in self.get_representatives_from_generation(self.generation, 1)])
            for individual in self.hall_of_fame:
                individual = individual.clone()
                # TODO: Figure out what this line was supposed to do (replace with clone to prevent overlap?)

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

    def tournament_selection(self):
        tournament = random.sample(self.population, 2)
        tournament.sort(key=self.calculate_diversity_fitness, reverse=True)
        return tournament[0]

    def generate_mutation(self):
        #parent = self.fitness_proportionate_selection()
        parent = self.tournament_selection()
        child = parent.clone()
        child.mutate()
        return child

    def generate_recombination(self):
        #parent = self.fitness_proportionate_selection()
        #donor = self.fitness_proportionate_selection()
        parent = self.tournament_selection()
        donor = self.tournament_selection()
        child = parent.clone()
        child.recombine(donor)
        return child
