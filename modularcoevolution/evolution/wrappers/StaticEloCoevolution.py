from modularcoevolution.evolution.wrappers.Coevolution import Coevolution
from modularcoevolution.evolution.wrappers.EloCoevolution import EloCoevolution

class StaticEloCoevolution(EloCoevolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_scores = dict()
        self.opponents = dict()
        
        self.deferred_evaluation_results = dict()
        self.elos_to_update = list()

    def start_generation(self):
        self.total_scores = dict()
        self.opponents = dict()
        self.deferred_evaluation_results = dict()
        self.elos_to_update = list()
        super().start_generation()

    def send_objectives(self, evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags=None,
                        defender_average_flags=None, attacker_inactive_objectives=None,
                        defender_inactive_objectives=None):
        #args = [evaluation_ID, attacker_objectives, defender_objectives]
        #kwargs = {"attacker_average_flags": attacker_average_flags,
        #                "defender_average_flags": defender_average_flags, "attacker_inactive_objectives": attacker_inactive_objectives,
        #                "defender_inactive_objectives": defender_inactive_objectives}

        self.update_objective_range(attacker_objectives, attacker_objectives)
        self.update_objective_range(defender_objectives, defender_objectives)

        attacker, defender = self.get_pair(evaluation_ID)
        attacker_ID = attacker.genotype.ID
        defender_ID = defender.genotype.ID
        for objective in list(attacker_objectives) + list(defender_objectives):
            if attacker_ID not in self.elos:
                self.elos[attacker_ID] = dict()
            if objective not in self.elos[attacker_ID]:
                self.elos[attacker_ID][objective] = 0
            if defender_ID not in self.elos:
                self.elos[defender_ID] = dict()
            if objective not in self.elos[defender_ID]:
                self.elos[defender_ID][objective] = 0
            if attacker_ID not in self.total_scores:
                self.total_scores[attacker_ID] = dict()
            if objective not in self.total_scores[attacker_ID]:
                self.total_scores[attacker_ID][objective] = 0
            if defender_ID not in self.total_scores:
                self.total_scores[defender_ID] = dict()
            if objective not in self.total_scores[defender_ID]:
                self.total_scores[defender_ID][objective] = 0
            if attacker_ID not in self.opponents:
                self.opponents[attacker_ID] = list()
            if defender_ID not in self.opponents:
                self.opponents[defender_ID] = list()
        
        self.opponents[attacker_ID].append(defender_ID)
        self.opponents[defender_ID].append(attacker_ID)
        self.elos_to_update.append(attacker_ID)
        self.elos_to_update.append(defender_ID)
        
        for objective, score in attacker_objectives.items():
            if self.max_objectives[objective] == self.min_objectives[objective]:
                normalized_score = 0.5
            else:
                normalized_score = (score - self.min_objectives[objective]) / (
                        self.max_objectives[objective] - self.min_objectives[objective])
                #normalized_score = normalized_score * 0.9 + 0.5 * 0.1
            self.total_scores[attacker_ID][objective] += normalized_score
            self.total_scores[defender_ID][objective] += 1 - normalized_score
        for objective, score in [(objective, score) for objective, score in defender_objectives.items() if objective not in attacker_objectives]:
            if self.max_objectives[objective] == self.min_objectives[objective]:
                normalized_score = 0.5
            else:
                normalized_score = (score - self.min_objectives[objective]) / (
                        self.max_objectives[objective] - self.min_objectives[objective])
                #normalized_score = normalized_score * 0.5 + 0.5 * 0.5
            self.total_scores[defender_ID][objective] += normalized_score
            self.total_scores[attacker_ID][objective] += 1 - normalized_score

        args = (evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags, defender_average_flags, attacker_inactive_objectives, defender_inactive_objectives)
        if evaluation_ID in self.remaining_evolution_evaluations:
            self.deferred_evaluation_results[evaluation_ID] = args
        else:
            super().send_objectives(*args)
            
        if len(self.deferred_evaluation_results) == len(self.remaining_evolution_evaluations):
            self.process_deferred_objectives()
    
    def calculate_static_elos(self):
        initial_rating = 0
        iteration_elos = {player_id: dict() for player_id in self.elos_to_update}
        for player in self.elos_to_update:
            iteration_elos[player] = {objective: initial_rating for objective in self.elos[player]}
        min_opponents = min([len(self.opponents[player]) for player in self.elos_to_update])
        if min_opponents == 1:
            iterations = 1
        else:
            iterations = 100
        for i in range(iterations):
            average_error = 0
            new_elos = dict()
            for player in self.elos_to_update:
                new_elos[player] = dict()
                for objective in self.elos[player]:
                    average_opponent_elo = sum([iteration_elos[opponent][objective] for opponent in self.opponents[player] if opponent in iteration_elos]) / len(
                        self.opponents[player])  # R_c
                    average_score = self.total_scores[player][objective] / len(self.opponents[player])
                    difference = self.expected_difference(average_score)
                    new_elo = average_opponent_elo * 0.9 + difference
                    average_error += abs(iteration_elos[player][objective] - new_elo) / len(self.elos_to_update)
                    new_elos[player][objective] = new_elo
            iteration_elos = new_elos
            #if i % 1 == 0:
            #   print(f"Average error: {average_error}")
            #   print({player_id: iteration_elos[player_id]["time remaining"] for player_id in iteration_elos})
            #   print(str(iteration_elos[0]["time remaining"]) + "\t" + str(iteration_elos[50]["time remaining"]))
            if average_error < 0.1:
                break
        self.elos.update(iteration_elos)

    def process_deferred_objectives(self):
        self.calculate_static_elos()

        for evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags, defender_average_flags, attacker_inactive_objectives, defender_inactive_objectives in self.deferred_evaluation_results.values():
            if attacker_average_flags is None:
                attacker_average_flags = dict()
            if defender_average_flags is None:
                defender_average_flags = dict()
            if attacker_inactive_objectives is None:
                attacker_inactive_objectives = list()
            if defender_inactive_objectives is None:
                defender_inactive_objectives = list()

            self.total_evaluations += 1
            attacker, defender = self.get_pair(evaluation_ID)
            attacker_ID = attacker.genotype.ID
            defender_ID = defender.genotype.ID

            attacker_objective_elos = {(objective + " elo"): self.elos[attacker_ID][objective] for objective in
                                       attacker_objectives}
            defender_objective_elos = {(objective + " elo"): self.elos[defender_ID][objective] for objective in
                                       defender_objectives}
            attacker_inactive_objectives.extend(attacker_objectives)
            defender_inactive_objectives.extend(defender_objectives)
            attacker_objectives.update(attacker_objective_elos)
            defender_objectives.update(defender_objective_elos)
            attacker_average_flags.update({objective: False for objective in attacker_objective_elos})
            defender_average_flags.update({objective: False for objective in defender_objective_elos})

            Coevolution.send_objectives(self, evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags, defender_average_flags, False, False, attacker_inactive_objectives, defender_inactive_objectives)

        min_opponents = min([len(self.opponents[player]) for player in self.elos_to_update])
        self.remaining_evolution_evaluations = list()
        self.elos_to_update = list()
        self.deferred_evaluation_results = dict()

        if min_opponents < self.evaluations_per_individual:
            self.add_additional_evaluations()



