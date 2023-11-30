from modularcoevolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator

import os


class EvolutionWrapper:  # TODO: Merge into common superclass with CoevolutionWrapper
    def __init__(self, generator: BaseEvolutionaryGenerator, num_generations, evaluations_per_individual=1, data_collector=None, log_subfolder=""):
        assert isinstance(generator, BaseEvolutionaryGenerator)
        self.generator = generator

        self.evaluation_ID_counter = 0
        self.evaluation_table = dict()
        self.remaining_evolution_evaluations = list()

        self.generation = 0
        self.num_generations = num_generations
        self.finalizing = False
        self.evaluations_per_individual = evaluations_per_individual

        if log_subfolder != "" and not log_subfolder.startswith("/"):
            log_subfolder = "/" + log_subfolder
        self.log_path = "Logs" + log_subfolder

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.result_log = open(self.log_path + "/resultLog.txt", "a+")
        self.solution_log = open(self.log_path + "/solutionLog.txt", "a+")
        if self.generation == 0:
            self.result_log.truncate(0)
            self.solution_log.truncate(0)

        self.data_collector = data_collector

        self.start_generation()

    def next_generation(self):
        if self.generation >= self.num_generations:
            self.finalizing = True
        if self.finalizing:
            raise EvolutionEndedException
        else:
            print("Starting next generation----------------------------------")
            self.generation += 1
            self.generator.next_generation(self.result_log, None)  # self.solution_log)

        self.remaining_evolution_evaluations.clear()
        self.start_generation()

    def start_generation(self):
        individuals = self.generator.get_individuals_to_test()
        for individual in individuals:
            for _ in range(self.evaluations_per_individual):
                evaluation_ID = self.claim_evaluation_ID()
                self.evaluation_table[evaluation_ID] = individual
                self.remaining_evolution_evaluations.append(evaluation_ID)

    def get_individual(self, evaluation_ID):
        return self.generator.build_agent_from_id(self.evaluation_table[evaluation_ID], active=True)

    def get_remaining_evaluations(self):
        remaining_evaluations = list()
        remaining_evaluations.extend(self.remaining_evolution_evaluations)
        return remaining_evaluations

    def claim_evaluation_ID(self):
        evaluation_ID = self.evaluation_ID_counter
        self.evaluation_ID_counter += 1
        return evaluation_ID

    def send_objectives(self, evaluation_ID, objectives, average_flags=None,
                        average_fitness=True, inactive_objectives=None):
        if average_flags is None:
            average_flags = dict()
        average_flags.update(
            {objective: True for objective in objectives if objective not in average_flags})
        if inactive_objectives is None:
            inactive_objectives = list()

        individual = self.get_individual(evaluation_ID)
        if self.data_collector is not None:
            individual_name = type(individual).agent_type_name
            self.data_collector.set_evaluation_data(evaluation_ID, {individual_name: individual.genotype.id},
                                                    {individual_name: objectives})

        individual_ID = individual.genotype.id
        self.generator.set_objectives(self.evaluation_table[evaluation_ID], objectives, average_flags=average_flags,
                                      average_fitness=average_fitness, evaluation_id=evaluation_ID,
                                      inactive_objectives=inactive_objectives)
        self.remaining_evolution_evaluations.remove(evaluation_ID)

    def check_generation_end(self):
        return self.agent_id >= self.generator.population_size - 1

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["result_log"]
        del state["solution_log"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.result_log = open(self.log_path + "/resultLog.txt", "a+")
        self.solution_log = open(self.log_path + "/solutionLog.txt", "a+")

class EvolutionEndedException(Exception):
    pass
