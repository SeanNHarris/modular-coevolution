from modularcoevolution.evolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator
from modularcoevolution.evolution.wrappers.BaseEvolutionWrapper import EvolutionEndedException

import math
import os
import random


# Runs the coevolutionary algorithm with a generic attacker and defender.
class Coevolution():
    def __init__(self, attacker_generator, defender_generator, num_generations, evaluations_per_individual,
                 tournament_evaluations=None, tournament_batch_size=4, tournament_ratio=1, data_collector=None,
                 log_subfolder=""):
        assert isinstance(attacker_generator, BaseEvolutionaryGenerator)
        assert isinstance(defender_generator, BaseEvolutionaryGenerator)
        self.attacker_generator = attacker_generator
        self.defender_generator = defender_generator
        self.current_attackers = list()
        self.current_defenders = list()

        self.evaluation_ID_counter = 0
        self.evaluation_table = dict()
        self.remaining_evolution_evaluations = list()
        self.remaining_tournament_evaluations = list()
        self.tournament_buffer = list()
        self.tournament_evaluations = tournament_evaluations
        if self.tournament_evaluations is None:
            self.tournament_evaluations = evaluations_per_individual
        self.tournament_evaluation_count = dict()
        self.tournament_batch_size = tournament_batch_size
        self.tournament_ratio = tournament_ratio
        self.tournament_objectives_attacker = dict()
        self.tournament_objectives_defender = dict()

        self.generation = 0
        self.num_generations = num_generations
        self.finalizing = False
        self.evaluations_per_individual = evaluations_per_individual  # Individual pairs will be selected until all have been tested at least this many times.
        self.completed_pairings = dict()
        self.opponents_this_generation = dict()
        
        if log_subfolder != "" and not log_subfolder.startswith("/"):
            log_subfolder = "/" + log_subfolder
        self.log_path = "Logs" + log_subfolder
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.result_log_attacker = open(self.log_path + "/resultLogAttacker.txt", "a+")
        self.solution_log_attacker = open(self.log_path + "/solutionLogAttacker.txt", "a+")
        self.result_log_defender = open(self.log_path + "/resultLogDefender.txt", "a+")
        self.solution_log_defender = open(self.log_path + "/solutionLogDefender.txt", "a+")
        self.tournament_data_attacker = open(self.log_path + "/tournamentDataAttacker.txt", "a+")
        self.tournament_data_defender = open(self.log_path + "/tournamentDataDefender.txt", "a+")
        if self.generation == 0:
            self.result_log_attacker.truncate(0)
            self.solution_log_attacker.truncate(0)
            self.result_log_defender.truncate(0)
            self.solution_log_defender.truncate(0)
            self.tournament_data_attacker.truncate(0)
            self.tournament_data_defender.truncate(0)

        self.data_collector = data_collector

        self.start_generation()

    # Causes the attacker and defender to both run their next generation code, and log the results of the last generation
    def next_generation(self):
        if self.finalizing:
            raise EvolutionEndedException

        print("Starting next generation----------------------------------")
        if self.generation >= self.num_generations:
            self.finalizing = True
        else:
            self.generation += 1
            self.attacker_generator.next_generation(self.result_log_attacker, None)#self.solution_log_attacker)
            self.defender_generator.next_generation(self.result_log_defender, None)#self.solution_log_defender)

        self.remaining_evolution_evaluations.clear()
        self.remaining_tournament_evaluations.clear()
        self.start_generation()

    def start_generation(self):
        if not self.finalizing:
            self.completed_pairings = dict()
            self.opponents_this_generation = dict()
            self.add_initial_evaluations()
            current_generation = self.generation - 1
            if self.tournament_ratio > 0 and current_generation % self.tournament_ratio == 0:
                tournament_representatives = math.ceil(math.sqrt(self.tournament_evaluations))
                for opponent_generation in range(0, self.generation, self.tournament_ratio):
                    recent_attacker_representatives = self.attacker_generator \
                        .get_representatives_from_generation(current_generation, tournament_representatives)
                    recent_defender_representatives = self.defender_generator \
                        .get_representatives_from_generation(current_generation, tournament_representatives)
                    past_attacker_representatives = self.attacker_generator \
                        .get_representatives_from_generation(opponent_generation, tournament_representatives)
                    past_defender_representatives = self.defender_generator \
                        .get_representatives_from_generation(opponent_generation, tournament_representatives)
                    vertical_pairs = self.generate_pairs(len(recent_attacker_representatives), len(past_defender_representatives),
                                                tournament_representatives)
                    horizontal_pairs = self.generate_pairs(len(past_attacker_representatives),
                                                         len(recent_defender_representatives),
                                                         tournament_representatives)
                    self.tournament_evaluation_count[(current_generation, opponent_generation)] = len(vertical_pairs)
                    self.tournament_evaluation_count[(opponent_generation, current_generation)] = len(horizontal_pairs)
                    for pair in vertical_pairs:
                        evaluation_ID = self.claim_evaluation_ID()
                        self.evaluation_table[evaluation_ID] = (
                            (current_generation, recent_attacker_representatives[pair[0]]),
                            (opponent_generation, past_defender_representatives[pair[1]]))
                        self.tournament_buffer.append(evaluation_ID)
                    if opponent_generation != current_generation:
                        for pair in horizontal_pairs:
                            reversed_evaluation_ID = self.claim_evaluation_ID()
                            self.evaluation_table[reversed_evaluation_ID] = (
                                (opponent_generation, past_attacker_representatives[pair[0]]),
                                (current_generation, recent_defender_representatives[pair[1]]))
                            self.tournament_buffer.append(reversed_evaluation_ID)
        while len(self.tournament_buffer) >= self.tournament_batch_size:
            for _ in range(self.tournament_batch_size):
                self.remaining_tournament_evaluations.append(self.tournament_buffer.pop(0))
        if self.finalizing:
            self.remaining_tournament_evaluations.extend(self.tournament_buffer)
            self.tournament_buffer.clear()

    def add_initial_evaluations(self):
        self.current_attackers = self.attacker_generator.get_individuals_to_test()
        self.current_defenders = self.defender_generator.get_individuals_to_test()
        pair_ids = self.generate_pairs(len(self.current_attackers), len(self.current_defenders), self.evaluations_per_individual)
        pairs = [(self.current_attackers[i], self.current_defenders[j]) for i, j in pair_ids]
        for pair in pairs:
            evaluation_ID = self.claim_evaluation_ID()
            self.evaluation_table[evaluation_ID] = pair
            self.remaining_evolution_evaluations.append(evaluation_ID)


    # Generates pairs of attacker and defender IDs
    #@numba.jit()
    def generate_pairs(self, size_1, size_2, pairs_per_individual, invalid_pairs=None, allow_repeats=False):
        if invalid_pairs is None:
            invalid_pairs = {}
        reverse = False
        lesser = size_1
        greater = size_2
        if lesser > greater:
            lesser, greater = greater, lesser
            reverse = True
        if pairs_per_individual > lesser:
            if allow_repeats:
                print("Warning: low population size will lead to repeat evaluations.")
            else:
                pairs_per_individual = lesser

        pairs = list()
        lesser_count = dict()

        for i in range(lesser):
            lesser_count[i] = 0

        for j in range(greater):
            valid = [i for i in range(lesser) if (i, j) not in pairs and (i, j) not in invalid_pairs]
            if len(valid) == 0:
                print("Failure in pair generation: No valid pairs.")
                return self.generate_pairs(size_1, size_2, pairs_per_individual, None, allow_repeats)
            random.shuffle(valid)
            valid.sort(key=lesser_count.get)
            #print([(i, lesser_count[i]) for i in valid])
            for p in range(pairs_per_individual):
                i = valid[p % len(valid)]  # Modulus in case of lower population than evaluations per individual
                pairs.append((i, j))
                lesser_count[i] += 1
        if max(lesser_count.values()) - min(lesser_count.values()) > 1:
            print("Failure in pair generation: Uneven evaluation pairings.")
            return self.generate_pairs(size_1, size_2, pairs_per_individual, None, allow_repeats)
        if reverse:
            pairs = [(j, i) for (i, j) in pairs]
        return pairs

    # Compare by highest standard deviation, highest objective, and least number of evaluations
    def compare_by_interest(self, individual_1, individual_2):
        standard_deviation_1 = max([individual_1.objective_statistics[objective]["standard deviation"] for objective in
                                    individual_1.objectives])
        standard_deviation_2 = max([individual_2.objective_statistics[objective]["standard deviation"] for objective in
                                    individual_2.objectives])
        highest_objective_1 = max(individual_1.objectives.values())
        highest_objective_2 = max(individual_2.objectives.values())
        individual_1_score = highest_objective_1 * standard_deviation_1 / \
                             len(self.opponents_this_generation[individual_1.ID])
        individual_2_score = highest_objective_2 * standard_deviation_2 / \
                             len(self.opponents_this_generation[individual_2.ID])
        return individual_1_score - individual_2_score

    def claim_evaluation_ID(self):
        evaluation_ID = self.evaluation_ID_counter
        self.evaluation_ID_counter += 1
        return evaluation_ID

    def get_remaining_evaluations(self):
        remaining_evaluations = list()
        remaining_evaluations.extend(self.remaining_evolution_evaluations)
        remaining_evaluations.extend(self.remaining_tournament_evaluations)
        return remaining_evaluations

    def get_pair(self, evaluation_ID):
        return self.attacker_generator.get_from_generation(*self.evaluation_table[evaluation_ID][0]), \
               self.defender_generator.get_from_generation(*self.evaluation_table[evaluation_ID][1])

    def send_objectives(self, evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags=None,
                        defender_average_flags=None, attacker_average_fitness=True, defender_average_fitness=True,
                        attacker_inactive_objectives=None, defender_inactive_objectives=None):
        if attacker_average_flags is None:
            attacker_average_flags = dict()
        attacker_average_flags.update(
            {objective: True for objective in attacker_objectives if objective not in attacker_average_flags})
        if defender_average_flags is None:
            defender_average_flags = dict()
        defender_average_flags.update(
            {objective: True for objective in defender_objectives if objective not in defender_average_flags})
        if attacker_inactive_objectives is None:
            attacker_inactive_objectives = list()
        if defender_inactive_objectives is None:
            defender_inactive_objectives = list()



        if self.data_collector is not None:
            attacker, defender = self.get_pair(evaluation_ID)
            if hasattr(attacker, "agent_type_name"):
                attacker_name = attacker.agent_type_name
            else:
                attacker_name = type(attacker).agent_type_name
            if hasattr(defender, "agent_type_name"):
                defender_name = defender.agent_type_name
            else:
                defender_name = type(defender).agent_type_name
            self.data_collector.set_evaluation_data(evaluation_ID,
                                                    {attacker_name: attacker.genotype.ID, defender_name: defender.genotype.ID},
                                                    {attacker_name: attacker_objectives, defender_name: defender_objectives})
        if evaluation_ID in self.remaining_evolution_evaluations:
            pair = self.evaluation_table[evaluation_ID]
            self.completed_pairings[pair] = self.completed_pairings.setdefault(pair, 0) + 1
            attacker, defender = self.get_pair(evaluation_ID)
            attacker_ID = attacker.genotype.ID
            defender_ID = defender.genotype.ID
            self.attacker_generator.set_objectives(self.evaluation_table[evaluation_ID][0][1], attacker_objectives,
                                                   average_flags=attacker_average_flags, average_fitness=attacker_average_fitness,
                                                   opponent=defender, evaluation_number=evaluation_ID,
                                                   inactive_objectives=attacker_inactive_objectives)
            self.defender_generator.set_objectives(self.evaluation_table[evaluation_ID][1][1], defender_objectives,
                                                   average_flags=defender_average_flags, average_fitness=defender_average_fitness,
                                                   opponent=attacker, evaluation_number=evaluation_ID,
                                                   inactive_objectives=defender_inactive_objectives)
            self.remaining_evolution_evaluations.remove(evaluation_ID)

            if attacker_ID not in self.opponents_this_generation:
                self.opponents_this_generation[attacker_ID] = set()
            self.opponents_this_generation[attacker_ID].add(defender_ID)
            if defender_ID not in self.opponents_this_generation:
                self.opponents_this_generation[defender_ID] = set()
            self.opponents_this_generation[defender_ID].add(attacker_ID)
        else:  # tournament
            attacker, defender = self.get_pair(evaluation_ID)
            if len(attacker_objectives) > 0:
                if self.attacker_generator.fitness_function is not None:
                    attacker_objectives["quality"] = self.attacker_generator.fitness_function(attacker_objectives)
                else:
                    attacker_objectives["quality"] = sum(attacker_objectives.values()) / len(attacker_objectives)
            if len(defender_objectives) > 0:
                if self.defender_generator.fitness_function is not None:
                    defender_objectives["quality"] = self.defender_generator.fitness_function(defender_objectives)
                else:
                    defender_objectives["quality"] = sum(defender_objectives.values()) / len(defender_objectives)

            attacker_generation = self.evaluation_table[evaluation_ID][0][0]
            defender_generation = self.evaluation_table[evaluation_ID][1][0]
            for individual_objectives, tournament_objectives in \
                    [(attacker_objectives, self.tournament_objectives_attacker), (defender_objectives, self.tournament_objectives_defender)]:
                for objective_name in individual_objectives:
                    if objective_name not in tournament_objectives:
                        tournament_objectives[objective_name] = dict()
                    if attacker_generation not in tournament_objectives[objective_name]:
                        tournament_objectives[objective_name][attacker_generation] = dict()
                    if defender_generation not in tournament_objectives[objective_name][attacker_generation]:
                        tournament_objectives[objective_name][attacker_generation][defender_generation] = 0
                    tournament_objectives[objective_name][attacker_generation][defender_generation] += \
                        individual_objectives[objective_name] / self.tournament_evaluation_count[(attacker_generation, defender_generation)]
                    if self.data_collector is not None:
                        self.data_collector.set_experiment_master_tournament_objective(objective_name, tournament_objectives[objective_name])

            self.remaining_tournament_evaluations.remove(evaluation_ID)
            self.write_tournament_data()

    def write_tournament_data(self):
        last_tournament = (self.num_generations - 1) - (self.num_generations - 1) % self.tournament_ratio
        for tournament_objectives, tournament_data in (
                (self.tournament_objectives_attacker, self.tournament_data_attacker),
                (self.tournament_objectives_defender, self.tournament_data_defender)):
            tournament_data.truncate(0)
            for objective_name in tournament_objectives:
                tournament_data.write("{0}\n".format(objective_name))
                for attacker_generation in range(last_tournament, -1, -self.tournament_ratio):
                    for defender_generation in range(0, self.num_generations, self.tournament_ratio):
                        if attacker_generation in tournament_objectives[objective_name] and defender_generation in \
                                tournament_objectives[objective_name][attacker_generation]:
                            tournament_data.write("{0} ".format(
                                tournament_objectives[objective_name][attacker_generation][defender_generation]))
                        else:
                            tournament_data.write("None ")
                    tournament_data.write("\n")
            tournament_data.flush()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["result_log_attacker"]
        del state["solution_log_attacker"]
        del state["result_log_defender"]
        del state["solution_log_defender"]
        del state["tournament_data_attacker"]
        del state["tournament_data_defender"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.result_log_attacker = open(self.log_path + "/resultLogAttacker.txt", "a+")
        self.solution_log_attacker = open(self.log_path + "/solutionLogAttacker.txt", "a+")
        self.result_log_defender = open(self.log_path + "/resultLogDefender.txt", "a+")
        self.solution_log_defender = open(self.log_path + "/solutionLogDefender.txt", "a+")
        self.tournament_data_attacker = open(self.log_path + "/tournamentDataAttacker.txt", "a+")
        self.tournament_data_defender = open(self.log_path + "/tournamentDataDefender.txt", "a+")
