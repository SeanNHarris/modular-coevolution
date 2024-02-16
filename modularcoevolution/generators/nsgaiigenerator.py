from modularcoevolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator
from modularcoevolution.genotypes.selfadaptivewrapper import SelfAdaptiveWrapper

import functools
import math
import random


# Note: these functions assume objective maximization
class NSGAIIGenerator(BaseEvolutionaryGenerator):
    def __init__(
            self,
            agent_class,
            population_name,
            initial_size,
            children_size,
            agent_parameters=None,
            genotype_parameters=None,
            mutation_fraction=0.25,
            recombination_fraction=0.75,
            tournament_size=2,
            seed=None,
            fitness_function=None,
            data_collector=None,
            copy_survivor_objectives=False,
            reevaluate_per_generation=True,
            using_hall_of_fame=True,
            compute_diversity=False,
            past_population_width=-1):
        super().__init__(agent_class=agent_class, population_name=population_name, initial_size=initial_size, agent_parameters=agent_parameters, genotype_parameters=genotype_parameters, seed=seed, fitness_function=fitness_function,
                         data_collector=data_collector, copy_survivor_objectives=copy_survivor_objectives, reevaluate_per_generation=reevaluate_per_generation, using_hall_of_fame=using_hall_of_fame, compute_diversity=compute_diversity, past_population_width=past_population_width)
        self.children_size = children_size
        self.mutation_fraction = mutation_fraction
        self.recombination_fraction = recombination_fraction
        self.tournament_size = tournament_size

        self.past_fronts = list()

    # Returns indices
    def get_representatives_from_generation(self, generation, amount, force=False):
        #pareto_front = self.past_fronts[generation][0]
        #if force:
        #    return [i % len(pareto_front) for i in range(amount)]
        #else:
        #    return range(min(amount, len(pareto_front)))
        # Indexes of the amount individuals with the highest quality.
        fitness_best = sorted(range(len(self.past_populations[generation])), key=lambda index: self.past_populations[generation][index].metrics["quality"], reverse=True)
        return fitness_best[:amount]

    def get_fitness(self, index):
        raise NotImplementedError("This is a multi-objective algorithm and does not store a single fitness.")

    def set_fitness(self, index, fitness, average=False):
        raise NotImplementedError("This is a multi-objective algorithm and does not store a single fitness.")

    def next_generation(self, result_log=None, agent_log=None):
        nondominating_fronts = nondominated_sort(self.population)
        crowding_distances = dict()
        for front in nondominating_fronts:
            crowding_distances.update(calculate_crowding_distances(front))
            
        survivors = list()
        truncated_front = 0
        while truncated_front < len(nondominating_fronts) and len(survivors) + len(nondominating_fronts[truncated_front]) <= self.initial_size:
            survivors.extend(nondominating_fronts[truncated_front])
            truncated_front += 1

        def crowded_comparison(individual_1, individual_2):
            domination = domination_comparison(individual_1, individual_2)
            if domination != 0:
                return domination
            else:
                return crowding_distances[individual_1] - crowding_distances[individual_2]

        self.population.sort(key=functools.cmp_to_key(crowded_comparison), reverse=True)
        for front in nondominating_fronts:
            front.sort(key=functools.cmp_to_key(crowded_comparison), reverse=True)

        if truncated_front < len(nondominating_fronts):  # False on the first generation, since everyone survives
            survivors.extend(nondominating_fronts[truncated_front][0:self.initial_size - len(survivors)])

        objective_names = self.population[0].objectives.keys()
        best_objectives = dict()
        average_nondominated_objectives = dict()
        average_objectives = dict()
        worst_nondominated_objectives = dict()
        for name in objective_names:
            best_objectives[name] = -math.inf
            average_nondominated_objectives[name] = 0
            average_objectives[name] = 0
            worst_nondominated_objectives[name] = math.inf
            for individual in nondominating_fronts[0]:
                best_objectives[name] = max(best_objectives[name], individual.objectives[name])
                average_nondominated_objectives[name] += individual.objectives[name]
                worst_nondominated_objectives[name] = min(worst_nondominated_objectives[name], individual.objectives[name])
            average_nondominated_objectives[name] /= len(nondominating_fronts[0])
            for individual in self.population:
                average_objectives[name] += individual.objectives[name]
            average_objectives[name] /= len(self.population)
        diversity = self.get_diversity(presorted=True)

        if result_log is not None:
            result_columns = ["Generation"]
            for name in objective_names:
                result_columns.extend([name + " best", name + " average nondominated", name + " average", name + " worst nondominated"])
            result_columns.extend(["Front sizes" + "-"*20, "diversity"])
            result_log_format = "".join(
                ["{{!s:<{0}.{1}}}".format(len(column) + 4, len(column)) for column in result_columns]) + "\n"
            if self.generation == 0:
                result_log.truncate(0)
                result_log.write(result_log_format.format(*result_columns))

            result = [self.generation]
            for name in objective_names:
                result.extend([best_objectives[name], average_nondominated_objectives[name], average_objectives[name], worst_nondominated_objectives[name]])
            result.extend([str([len(front) for front in nondominating_fronts]), diversity])
            result_log.write(result_log_format.format(*result))
            result_log.flush()

        if agent_log is not None:
            agent_columns = ["ID", *objective_names, "Novelty", "Parent IDs", "Evaluations, opponents, and fitnesses",
                             "Genotype"]
            agent_log_format = "".join(["{{!s:<{0}}}".format(len(column) + 4) for column in agent_columns]) + "\n"
            if self.generation == 0:
                agent_log.truncate(0)
            agent_log.write("---Generation {}---\n".format(self.generation))
            agent_log.write(agent_log_format.format(*agent_columns))

            for individual in self.population:
                parent_string = "({})".format(individual.creation_method)
                if len(individual.parent_ids) == 1:
                    parent_string = "{} ({})".format(individual.parent_ids[0], individual.creation_method)
                elif len(individual.parent_ids) == 2:
                    parent_string = "{}, {} ({})".format(individual.parent_ids[0], individual.parent_ids[1],
                                                         individual.creation_method)

                evaluation_list = self.evaluation_lists[individual.id]
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
                    agent_log_format.format(individual.id, *individual.objectives.values(), individual.metrics["novelty"], parent_string,
                                            evaluation_string, str(individual)))
            agent_log.flush()

        if self.data_collector is not None:
            population_IDs = [individual.id for individual in self.population]
            objectives = dict()

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
            
            for name in objective_names:
                objectives[name].update({"average nondominated": average_nondominated_objectives[name], "worst nondominated": worst_nondominated_objectives[name]})
            front_members = [[individual.id for individual in front] for front in nondominating_fronts]
            metrics.update({"diversity": diversity, "front sizes": [len(front) for front in nondominating_fronts], "front members": front_members})
            self.data_collector.set_generation_data(self.population_name, self.generation, population_IDs, objectives, metrics)

        next_generation = list()
        next_generation_set = set()
        for i in range(self.initial_size):
            survivor = self.population[i].clone(copy_objectives=self.copy_survivor_objectives)
            next_generation.append(survivor)
            next_generation_set.add(survivor)
        
        num_mutation = int(math.ceil(self.mutation_fraction * self.children_size))
        num_recombination = int(math.floor(self.recombination_fraction * self.children_size))  # TODO: Pull all this duplicated code into a BaseEvolutionryGenerator abstract class
        
        for i in range(self.children_size):
            unique = False
            child = None
            while not unique:
                if i < num_mutation or (isinstance(child, SelfAdaptiveWrapper) and random.random() <
                                        child.self_adaptive_parameters["mutation rate"]):
                    parent = tournament_selection(survivors, crowded_comparison, self.tournament_size)
                    child = parent.clone()
                    child.mutate()
                else:
                    parent1 = tournament_selection(survivors, crowded_comparison, self.tournament_size)
                    parent2 = tournament_selection(survivors, crowded_comparison, self.tournament_size)
                    child = parent1.clone()
                    child.recombine(parent2)
                if hash(child) not in next_generation_set:
                    unique = True
            next_generation.append(child)
            next_generation_set.add(hash(child))

        self.past_populations.append(self.population[:self.past_population_width])
        self.past_fronts.append(nondominating_fronts)
        self.population = next_generation
        for genotype in self.population:
            self.genotypes_by_id[genotype.id] = genotype
        self.population_size = self.initial_size + self.children_size
        if self.using_hall_of_fame:
            self.hall_of_fame.extend([self.population[i] for i in self.get_representatives_from_generation(self.generation, 1)])
            self.hall_of_fame = [individual.clone() for individual in self.hall_of_fame]

        self.generation += 1


def nondominated_sort(population):
    dominated_sets = {individual: list() for individual in population}
    domination_counters = {individual: 0 for individual in population}
    nondominating_fronts = [list()]
    for individual in population:
        for other in population:
            domination = domination_comparison(individual, other)
            if domination > 0:
                dominated_sets[individual].append(other)
            elif domination < 0:
                domination_counters[individual] += 1
        if domination_counters[individual] == 0:
            nondominating_fronts[0].append(individual)

    front = 0
    while len(nondominating_fronts[front]) > 0:
        nondominating_fronts.append(list())
        for dominator in nondominating_fronts[front]:
            for dominated in dominated_sets[dominator]:
                domination_counters[dominated] -= 1
                if domination_counters[dominated] == 0:
                    nondominating_fronts[front + 1].append(dominated)
        front += 1
    nondominating_fronts.pop(-1)
    return nondominating_fronts


def calculate_crowding_distances(front_population):
    objective_count = len(front_population[0].get_active_objectives())
    crowding_distances = {individual: 0 for individual in front_population}
    for objective in front_population[0].get_active_objectives():
        sorted_population = list(front_population)
        sorted_population.sort(key=lambda individual: individual.objectives[objective])
        crowding_distances[sorted_population[0]] = math.inf
        objective_min = sorted_population[0].objectives[objective]
        crowding_distances[sorted_population[-1]] = math.inf
        objective_max = sorted_population[-1].objectives[objective]

        for i in range(1, len(sorted_population) - 1):
            individual = sorted_population[i]
            if objective_max == objective_min:
                crowding_distances[individual] = math.inf  # Same as the boundaries?
            else:
                crowding_distances[individual] += sorted_population[i+1].objectives[objective] - sorted_population[i-1].objectives[objective] / (objective_max - objective_min)
    return crowding_distances


def domination_comparison(individual_1, individual_2):
    objective_count = len(individual_1.get_active_objectives())
    greater_than_count = 0
    equal_count = 0
    for objective in individual_1.get_active_objectives():
        if individual_1.objectives[objective] < individual_2.objectives[objective]:
            greater_than_count -= 1
        elif individual_1.objectives[objective] > individual_2.objectives[objective]:
            greater_than_count += 1
        else:
            equal_count += 1
    if equal_count == objective_count:
        return 0
    elif greater_than_count - equal_count == -objective_count:
        return -1
    elif greater_than_count + equal_count == objective_count:
        return 1
    else:
        return 0


def tournament_selection(population, comparison, tournament_size):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=functools.cmp_to_key(comparison), reverse=True)
    return tournament[0]