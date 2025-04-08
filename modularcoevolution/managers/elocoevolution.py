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

from modularcoevolution.managers.similarstrengthcoevolution import SimilarStrengthCoevolution

#import munkres

import math


# if TYPE_CHECKING:


class EloCoevolution(SimilarStrengthCoevolution):
    def __init__(self, *args, k_factor=40, elo_pairing=True, elo_ranking=False, **kwargs):
        self.k_factor = k_factor
        self.elo_pairing = elo_pairing
        self.elo_ranking = elo_ranking

        super().__init__(*args, **kwargs)

        self.total_evaluations = 0

    # Expected score for the first listed player
    def expected_score(self, rating_1, rating_2):
        return 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating_2 - rating_1) / 400))

    def expected_difference(self, score_1, perfect_difference=1000):
        if score_1 == 0:
            return -perfect_difference
        if score_1 == 1:
            return perfect_difference
        return 200 * math.log(score_1 / (1 - score_1), math.sqrt(10))

    def calculate_elo_update(self, rating_1, rating_2, score_1, k_factor=40):
        score_2 = 1 - score_1
        expected_score_1 = self.expected_score(rating_1, rating_2)
        expected_score_2 = 1 - expected_score_1
        rating_1_update = rating_1 + k_factor * (score_1 - expected_score_1)
        rating_2_update = rating_2 + k_factor * (score_2 - expected_score_2)
        return rating_1_update, rating_2_update, score_1 - expected_score_1
    
    def update_score(self, player_1_id, player_2_id, objective, normalized_score, k_factor=40):
        self.ratings[player_1_id][objective], self.ratings[player_2_id][
            objective], score_deviation = self.calculate_elo_update(self.ratings[player_1_id][objective],
                                                                    self.ratings[player_2_id][objective],
                                                                    normalized_score, self.k_factor)

    def calculate_ratings(self, ratings_to_update):
        for evaluation_id in self.scored_pairings:
            attacker_id, defender_id = self.evaluation_table[evaluation_id]

            for main_evaluation_objectives, main_id, secondary_id in [
                (self.evaluation_objectives_attacker, attacker_id, defender_id),
                (self.evaluation_objectives_defender, defender_id, attacker_id)
            ]:
                for objective, score in main_evaluation_objectives[evaluation_id].items():
                    if objective not in self.min_objectives or score < self.min_objectives[objective]:
                        self.min_objectives[objective] = score
                    if objective not in self.max_objectives or score > self.max_objectives[objective]:
                        self.max_objectives[objective] = score
                    if self.max_objectives[objective] == self.min_objectives[objective]:
                        normalized_score = 0.5
                    else:
                        normalized_score = (score - self.min_objectives[objective]) / (
                                self.max_objectives[objective] - self.min_objectives[objective])

                    self.update_score(main_id, secondary_id, objective, normalized_score, self.k_factor)
