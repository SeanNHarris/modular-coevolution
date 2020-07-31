from Evolution.Coevolution import *
from Evolution.EvolutionGenerator import *

import munkres


class EloCoevolution(Coevolution):
    def __init__(self, *args, k_factor=40, elo_pairing=True, elo_ranking=False, **kwargs):
        self.k_factor = k_factor
        self.elo_pairing = elo_pairing
        self.elo_ranking = elo_ranking

        super().__init__(*args, **kwargs)

        self.elos = dict()
        self.min_objectives = dict()
        self.max_objectives = dict()

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
        self.elos[player_1_id][objective], self.elos[player_2_id][
            objective], score_deviation = self.calculate_elo_update(self.elos[player_1_id][objective],
                                                                    self.elos[player_2_id][objective],
                                                                    normalized_score, self.k_factor)
        

    def update_objective_range(self, min_objectives, max_objectives):
        for objective, value in min_objectives.items():
            if objective not in self.min_objectives:
                self.min_objectives[objective] = value
            else:
                self.min_objectives[objective] = min(value, self.min_objectives[objective])
        for objective, value in max_objectives.items():
            if objective not in self.max_objectives:
                self.max_objectives[objective] = value
            else:
                self.max_objectives[objective] = max(value, self.max_objectives[objective])

    # TODO: Handle tournaments
    def send_objectives(self, evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags=None,
                        defender_average_flags=None, attacker_inactive_objectives=None,
                        defender_inactive_objectives=None):
        self.update_objective_range(attacker_objectives, attacker_objectives)
        self.update_objective_range(defender_objectives, defender_objectives)
        if attacker_average_flags is None:
            attacker_average_flags = dict()
        if defender_average_flags is None:
            defender_average_flags = dict()
        if attacker_inactive_objectives is None:
            attacker_inactive_objectives = list()
        if defender_inactive_objectives is None:
            defender_inactive_objectives = list()
        if evaluation_ID in self.remaining_evolution_evaluations:
            self.total_evaluations += 1
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

            for objective, score in attacker_objectives.items():
                if objective not in self.min_objectives or score < self.min_objectives[objective]:
                    self.min_objectives[objective] = score
                if objective not in self.max_objectives or score > self.max_objectives[objective]:
                    self.max_objectives[objective] = score
                if self.max_objectives[objective] == self.min_objectives[objective]:
                    normalized_score = 0.5
                else:
                    normalized_score = (score - self.min_objectives[objective]) / (
                        self.max_objectives[objective] - self.min_objectives[objective])

                self.update_score(attacker_ID, defender_ID, objective, normalized_score, self.k_factor)

            for objective, score in defender_objectives.items():
                if objective not in self.min_objectives or score < self.min_objectives[objective]:
                    self.min_objectives[objective] = min(score, 0)
                if objective not in self.max_objectives or score > self.max_objectives[objective]:
                    self.max_objectives[objective] = max(score, 100)
                if self.max_objectives[objective] == self.min_objectives[objective]:
                    normalized_score = 0.5
                else:
                    normalized_score = (score - self.min_objectives[objective]) / (
                            self.max_objectives[objective] - self.min_objectives[objective])

                self.update_score(defender_ID, attacker_ID, objective, normalized_score, self.k_factor)

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

        super().send_objectives(evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags,
                                defender_average_flags, False, False, attacker_inactive_objectives, defender_inactive_objectives)

        if len(self.remaining_evolution_evaluations) == 0 and self.total_evaluations < self.evaluations_per_individual * max(self.attacker_generator.population_size, self.defender_generator.population_size):
            self.add_additional_evaluations()

    def add_initial_evaluations(self):
        attackers = self.attacker_generator.get_individuals_to_test()
        defenders = self.defender_generator.get_individuals_to_test()
        if self.elo_pairing:
            pair_ids = self.generate_pairs(len(attackers), len(defenders), 1)
        else:
            pair_ids = self.generate_pairs(len(attackers), len(defenders), self.evaluations_per_individual)
        pairs = [(attackers[i], defenders[j]) for i, j in pair_ids]
        for pair in pairs:
            evaluation_ID = self.claim_evaluation_ID()
            self.evaluation_table[evaluation_ID] = pair
            self.remaining_evolution_evaluations.append(evaluation_ID)

        self.total_evaluations = 0

    def add_additional_evaluations(self):
        attackers = self.attacker_generator.get_individuals_to_test()
        defenders = self.defender_generator.get_individuals_to_test()
        #pairs = self.generate_matchmaking_pairs(attackers, defenders)
        pairs = self.generate_matchmaking_pairs_local_search(attackers, defenders)
        for attacker_index, defender_index in pairs:
            evaluation_ID = self.claim_evaluation_ID()
            self.evaluation_table[evaluation_ID] = (attackers[attacker_index], defenders[defender_index])
            self.remaining_evolution_evaluations.append(evaluation_ID)

    def generate_matchmaking_pairs_local_search(self, attackers, defenders):
        attacker_ids = dict()
        for attacker in attackers:
            attacker_ids[attacker] = self.attacker_generator.get_from_generation(*attacker).genotype.ID
        defender_ids = dict()
        for defender in defenders:
            defender_ids[defender] = self.defender_generator.get_from_generation(*defender).genotype.ID

        '''min_attacker_elo = 1000000
        max_attacker_elo = -1000000
        min_defender_elo = 1000000
        max_defender_elo = -1000000'''
        
        attacker_rankings = dict()
        for objective in self.elos[attacker_ids[attackers[0]]]:
            attacker_rankings[objective] = list(range(len(attackers)))
            attacker_rankings[objective].sort(key=lambda a: self.elos[attacker_ids[attackers[a]]][objective], reverse=True)
            attacker_rankings[objective] = [attacker_rankings[objective].index(i) for i in range(len(attacker_rankings[objective]))]
        defender_rankings = dict()
        for objective in self.elos[defender_ids[defenders[0]]]:
            defender_rankings[objective] = list(range(len(defenders)))
            defender_rankings[objective].sort(key=lambda d: self.elos[defender_ids[defenders[d]]][objective], reverse=True)
            defender_rankings[objective] = [defender_rankings[objective].index(i) for i in range(len(defender_rankings[objective]))]

        if not self.elo_ranking:
            def get_elo_distance(attacker, defender):  # Direct distance
                attacker_ID = self.attacker_generator.get_from_generation(*attacker).genotype.ID
                defender_ID = self.defender_generator.get_from_generation(*defender).genotype.ID
                if (attacker, defender) in self.completed_pairings:
                    return float("inf")
                sum_of_squares = 0
                for objective in self.elos[attacker_ID]:
                    sum_of_squares += (self.elos[attacker_ID][objective] - self.elos[defender_ID][
                        objective]) ** 2
                distance = math.sqrt(sum_of_squares)
                return distance
        else:
            def get_elo_distance(attacker, defender):  # Rank distance, preventing the failures that occur when the populations have heavily different mean ratings
                attacker_ID = self.attacker_generator.get_from_generation(*attacker).genotype.ID
                defender_ID = self.defender_generator.get_from_generation(*defender).genotype.ID
                if (attacker, defender) in self.completed_pairings:
                    return float("inf")
                sum_of_squares = 0
                for objective in self.elos[attacker_ID]:
                    sum_of_squares += (attacker_rankings[objective][attackers.index(attacker)] - defender_rankings[objective][defenders.index(defender)]) ** 2
                distance = math.sqrt(sum_of_squares)
                return distance

        invalid_pairs = list()
        distances = dict()
        for attacker, defender in self.completed_pairings:
            a = attackers.index(attacker)
            d = defenders.index(defender)
            invalid_pairs.append((a, d))

        pairs = self.generate_pairs(len(attackers), len(defenders), 1, invalid_pairs=invalid_pairs)
        #pair_attackers = [attacker for attacker, _ in pairs]
        #pair_defenders = [defender for _, defender in pairs]
        #print(sorted(pair_attackers))
        #print(sorted(pair_defenders))
        opponents = dict()
        for attacker, defender in pairs:
            opponents[attacker] = defender
        #distance_sum = sum([distances.setdefault((attacker, defender), self.get_elo_distance(attackers[attacker], defenders[defender])) for attacker, defender in opponents.items()])
        #print("Random distance of {}".format(distance_sum))
        improvement_steps = 0
        steps_since_improvement = 0
        while steps_since_improvement < 200:
            pair_1, pair_2 = random.choices(opponents, k=2)
            distance_1 = distances.setdefault((pair_1, opponents[pair_1]), get_elo_distance(attackers[pair_1], defenders[opponents[pair_1]]))
            distance_2 = distances.setdefault((pair_2, opponents[pair_2]), get_elo_distance(attackers[pair_2], defenders[opponents[pair_2]]))
            original_distance = distance_1 + distance_2
            #new_distance_sum = distance_sum - distance_1 - distance_2
            opponents[pair_1], opponents[pair_2] = opponents[pair_2], opponents[pair_1]
            distance_1 = distances.setdefault((pair_1, opponents[pair_1]), get_elo_distance(attackers[pair_1], defenders[opponents[pair_1]]))
            distance_2 = distances.setdefault((pair_2, opponents[pair_2]), get_elo_distance(attackers[pair_2], defenders[opponents[pair_2]]))
            new_distance = distance_1 + distance_2
            #new_distance_sum = new_distance_sum + distance_1 + distance_2
            if new_distance < original_distance:
                #distance_sum = new_distance_sum
                improvement_steps += 1
                steps_since_improvement = 0
            else:
                opponents[pair_1], opponents[pair_2] = opponents[pair_2], opponents[pair_1]
                steps_since_improvement += 1
        #print("Final distance of {0} after {1} improvements".format(distance_sum, improvement_steps))
        print("Hill-climber made {0} improvements.".format(improvement_steps))
        pairs = [(attacker, defender) for attacker, defender in opponents.items()]
        return pairs
            

    def generate_matchmaking_pairs(self, attackers, defenders):
        distances = list()
        for attacker in attackers:
            distances.append(list())
            attacker_ID = self.attacker_generator.get_from_generation(*attacker).genotype.ID
            for defender in defenders:
                defender_ID = self.defender_generator.get_from_generation(*defender).genotype.ID
                if (attacker, defender) in self.completed_pairings:
                    distances[-1].append(float("inf"))
                    continue
                sum_of_squares = 0
                for objective in self.elos[attacker_ID]:
                    sum_of_squares += (self.elos[attacker_ID][objective] - self.elos[defender_ID][objective]) ** 2
                distances[-1].append(math.sqrt(sum_of_squares))

        pairs = munkres.Munkres().compute(distances)
        distance_sum = sum([distances[attacker][defender] for attacker, defender in pairs])
        print("Ideal distance of {}".format(distance_sum))
        return pairs
