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

from modularcoevolution.managers.elocoevolution import EloCoevolution


class StaticEloCoevolution(EloCoevolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_scores = dict()

    def start_generation(self):
        self.total_scores = dict()
        super().start_generation()

    def preprocess_send_objectives(self, evaluation_id, attacker_objectives, defender_objectives,
                                   attacker_average_flags=None,
                                   defender_average_flags=None, attacker_average_fitness=True,
                                   defender_average_fitness=True,
                                   attacker_inactive_objectives=None, defender_inactive_objectives=None):
        super().preprocess_send_objectives(evaluation_id, attacker_objectives, defender_objectives,
                                           attacker_average_flags,
                                           defender_average_flags, attacker_average_fitness, defender_average_fitness,
                                           attacker_inactive_objectives, defender_inactive_objectives)

        attacker_id, defender_id = self.evaluation_table[evaluation_id]
        if attacker_id not in self.total_scores:
            self.total_scores[attacker_id] = {objective: 0 for objective in attacker_objectives}
        if defender_id not in self.total_scores:
            self.total_scores[defender_id] = {objective: 0 for objective in defender_objectives}

        for objective, score in attacker_objectives.items():
            if self.max_objectives[objective] == self.min_objectives[objective]:
                normalized_score = 0.5
            else:
                normalized_score = (score - self.min_objectives[objective]) / (
                        self.max_objectives[objective] - self.min_objectives[objective])
                #normalized_score = normalized_score * 0.9 + 0.5 * 0.1
            self.total_scores[attacker_id][objective] += normalized_score
            self.total_scores[defender_id][objective] += 1 - normalized_score
        for objective, score in [(objective, score) for objective, score in defender_objectives.items() if objective not in attacker_objectives]:
            if self.max_objectives[objective] == self.min_objectives[objective]:
                normalized_score = 0.5
            else:
                normalized_score = (score - self.min_objectives[objective]) / (
                        self.max_objectives[objective] - self.min_objectives[objective])
                #normalized_score = normalized_score * 0.5 + 0.5 * 0.5
            self.total_scores[defender_id][objective] += normalized_score
            self.total_scores[attacker_id][objective] += 1 - normalized_score

    def calculate_ratings(self, ratings_to_update):
        initial_rating = 0
        iteration_elos = {player_id: dict() for player_id in ratings_to_update}
        for player in ratings_to_update:
            iteration_elos[player] = {objective: initial_rating for objective in self.ratings[player]}
        min_opponents = min([len(self.opponents_this_generation[player]) for player in ratings_to_update])
        if min_opponents == 1:
            iterations = 1
        else:
            iterations = 100
        for i in range(iterations):
            average_error = 0
            new_elos = dict()
            for player in ratings_to_update:
                new_elos[player] = dict()
                for objective in self.ratings[player]:
                    average_opponent_elo = sum([iteration_elos[opponent][objective] for opponent in self.opponents_this_generation[player] if opponent in iteration_elos]) / len(
                        self.opponents_this_generation[player])  # R_c
                    average_score = self.total_scores[player][objective] / len(self.opponents_this_generation[player])
                    difference = self.expected_difference(average_score)
                    new_elo = average_opponent_elo * 0.9 + difference
                    average_error += abs(iteration_elos[player][objective] - new_elo) / len(ratings_to_update)
                    new_elos[player][objective] = new_elo
            iteration_elos = new_elos
            #if i % 1 == 0:
            #   print(f"Average error: {average_error}")
            #   print({player_id: iteration_elos[player_id]["time remaining"] for player_id in iteration_elos})
            #   print(str(iteration_elos[0]["time remaining"]) + "\t" + str(iteration_elos[50]["time remaining"]))
            if average_error < 0.1:
                break
        self.ratings.update(iteration_elos)
