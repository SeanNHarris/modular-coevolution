from Evolution.Coevolution import *
from Evolution.EvolutionGenerator import *

from abc import ABCMeta, abstractmethod


class SimilarStrengthCoevolution(Coevolution, metaclass=ABCMeta):
    def __init__(self, *args, enable_pairing=True, use_rank=False, **kwargs):
        self.enable_pairing = enable_pairing
        self.use_rank = use_rank

        super().__init__(*args, **kwargs)

        self.ratings = dict()
        self.min_objectives = dict()
        self.max_objectives = dict()

        self.scores_per_opponent = dict()  # ID -> opponent ID -> list of objective sets

        self.deferred_evaluation_results = dict()
        self.ratings_to_update = list()
        self.scored_pairings = list()

        self.total_evaluations = 0

    def start_generation(self):
        self.scores_per_opponent = dict()
        self.deferred_evaluation_results = dict()
        self.ratings_to_update = list()
        self.scored_pairings = list()
        super().start_generation()

    def next_generation(self):
        print("Attacker population:")
        for attacker in self.attacker_generator.population:
            objectives = list(attacker.objectives.values())
            print(f"{objectives[0]}, {objectives[1]}")
        print("Defender population:")
        for defender in self.defender_generator.population:
            objectives = list(defender.objectives.values())
            print(f"{objectives[0]}, {objectives[1]}")
        super().next_generation()

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
        # args = [evaluation_ID, attacker_objectives, defender_objectives]
        # kwargs = {"attacker_average_flags": attacker_average_flags,
        #                "defender_average_flags": defender_average_flags, "attacker_inactive_objectives": attacker_inactive_objectives,
        #                "defender_inactive_objectives": defender_inactive_objectives}
        self.update_objective_range(attacker_objectives, attacker_objectives)
        self.update_objective_range(defender_objectives, defender_objectives)

        attacker, defender = self.get_pair(evaluation_ID)
        attacker_ID = attacker.genotype.ID
        defender_ID = defender.genotype.ID
        for objective in list(attacker_objectives) + list(defender_objectives):
            if attacker_ID not in self.ratings:
                self.ratings[attacker_ID] = dict()
            if objective not in self.ratings[attacker_ID]:
                self.ratings[attacker_ID][objective] = 0
            if defender_ID not in self.ratings:
                self.ratings[defender_ID] = dict()
            if objective not in self.ratings[defender_ID]:
                self.ratings[defender_ID][objective] = 0
            if attacker_ID not in self.scores_per_opponent:
                self.scores_per_opponent[attacker_ID] = dict()
            if defender_ID not in self.scores_per_opponent[attacker_ID]:
                self.scores_per_opponent[attacker_ID][defender_ID] = dict()
            if objective not in self.scores_per_opponent[attacker_ID][defender_ID]:
                self.scores_per_opponent[attacker_ID][defender_ID][objective] = list()
            if defender_ID not in self.scores_per_opponent:
                self.scores_per_opponent[defender_ID] = dict()
            if attacker_ID not in self.scores_per_opponent[defender_ID]:
                self.scores_per_opponent[defender_ID][attacker_ID] = dict()
            if objective not in self.scores_per_opponent[defender_ID][attacker_ID]:
                self.scores_per_opponent[defender_ID][attacker_ID][objective] = list()

        for objective, score in attacker_objectives.items():
            self.scores_per_opponent[attacker_ID][defender_ID][objective].append(score)
            if objective not in defender_objectives:
                self.scores_per_opponent[defender_ID][attacker_ID][objective].append(0)
        for objective, score in defender_objectives.items():
            self.scores_per_opponent[defender_ID][attacker_ID][objective].append(score)
            if objective not in attacker_objectives:
                self.scores_per_opponent[attacker_ID][defender_ID][objective].append(0)


        args = (evaluation_ID, attacker_objectives, defender_objectives, attacker_average_flags, defender_average_flags,
                attacker_inactive_objectives, defender_inactive_objectives)
        if evaluation_ID in self.remaining_evolution_evaluations:
            self.ratings_to_update.append(attacker_ID)
            self.ratings_to_update.append(defender_ID)
            self.scored_pairings.append(evaluation_ID)
            self.deferred_evaluation_results[evaluation_ID] = args
        else:
            super().send_objectives(*args)

        if len(self.deferred_evaluation_results) == len(self.remaining_evolution_evaluations) > 0:
            self.process_deferred_objectives()

    @abstractmethod
    def calculate_ratings(self, ratings_to_update):
        pass

    def process_deferred_objectives(self):
        self.calculate_ratings(self.ratings_to_update)

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

            attacker_objective_ratings = {(objective + " rating"): self.ratings[attacker_ID][objective] for objective in
                                       attacker_objectives}
            defender_objective_ratings = {(objective + " rating"): self.ratings[defender_ID][objective] for objective in
                                       defender_objectives}
            attacker_inactive_objectives.extend(attacker_objectives)
            defender_inactive_objectives.extend(defender_objectives)
            attacker_objectives.update(attacker_objective_ratings)
            defender_objectives.update(defender_objective_ratings)
            attacker_average_flags.update({objective: False for objective in attacker_objective_ratings})
            defender_average_flags.update({objective: False for objective in defender_objective_ratings})

            Coevolution.send_objectives(self, evaluation_ID, attacker_objectives, defender_objectives,
                                        attacker_average_flags, defender_average_flags, False, False,
                                        attacker_inactive_objectives, defender_inactive_objectives)

        min_opponents = min([len(self.opponents_this_generation[player]) for player in self.ratings_to_update])
        self.remaining_evolution_evaluations = list()
        self.ratings_to_update = list()
        self.deferred_evaluation_results = dict()

        if min_opponents < self.evaluations_per_individual:
            self.add_additional_evaluations()

    def add_initial_evaluations(self):
        self.current_attackers = self.attacker_generator.get_individuals_to_test()
        self.current_defenders = self.defender_generator.get_individuals_to_test()
        if self.enable_pairing:
            pair_ids = self.generate_pairs(len(self.current_attackers), len(self.current_defenders), 1)
        else:
            pair_ids = self.generate_pairs(len(self.current_attackers), len(self.current_defenders), self.evaluations_per_individual)
        pairs = [(self.current_attackers[i], self.current_defenders[j]) for i, j in pair_ids]
        for pair in pairs:
            evaluation_ID = self.claim_evaluation_ID()
            self.evaluation_table[evaluation_ID] = pair
            self.remaining_evolution_evaluations.append(evaluation_ID)

        self.total_evaluations = 0

    def add_additional_evaluations(self):
        # pairs = self.generate_matchmaking_pairs(attackers, defenders)
        pairs = self.generate_matchmaking_pairs_greedy(self.current_attackers, self.current_defenders)
        for attacker_index, defender_index in pairs:
            evaluation_ID = self.claim_evaluation_ID()
            self.evaluation_table[evaluation_ID] = (self.current_attackers[attacker_index], self.current_defenders[defender_index])
            self.remaining_evolution_evaluations.append(evaluation_ID)

    def generate_matchmaking_pairs_greedy(self, attackers, defenders):
        # Select the closest valid partner to each individual, in decreasing order of fitness

        sorted_attackers = sorted(attackers, key=lambda attacker: self.attacker_generator.get_from_generation(*attacker).genotype.fitness, reverse=True)
        sorted_defenders = sorted(defenders, key=lambda defender: self.defender_generator.get_from_generation(*defender).genotype.fitness, reverse=True)

        pairs = list()
        while len(sorted_attackers) > 0 and len(sorted_defenders) > 0:
            opponent_index = 0
            while True:
                if (sorted_attackers[0], sorted_defenders[opponent_index]) not in self.completed_pairings:
                    pairs.append((sorted_attackers[0], sorted_defenders[opponent_index]))
                    sorted_attackers.pop(0)
                    sorted_defenders.pop(opponent_index)
                    break
                elif opponent_index > 0 and (sorted_attackers[opponent_index], sorted_defenders[0]) not in self.completed_pairings:
                    pairs.append((sorted_attackers[opponent_index], sorted_defenders[0]))
                    sorted_attackers.pop(opponent_index)
                    sorted_defenders.pop(0)
                    break
                else:
                    opponent_index += 1
                    if opponent_index >= len(sorted_attackers) or opponent_index >= len(sorted_defenders):
                        pairs.append((sorted_attackers[0], sorted_defenders[0]))
                        sorted_attackers.pop(0)
                        sorted_defenders.pop(0)
                        break
        return [(attackers.index(attacker), defenders.index(defender)) for attacker, defender in pairs]

    def generate_matchmaking_pairs_local_search(self, attackers, defenders):
        alternate_pairs = self.generate_matchmaking_pairs_greedy(attackers, defenders)  # TODO: REMOVE AFTER TESTING
        attacker_ids = dict()
        for attacker in attackers:
            attacker_ids[attacker] = self.attacker_generator.get_from_generation(*attacker).genotype.ID
        defender_ids = dict()
        for defender in defenders:
            defender_ids[defender] = self.defender_generator.get_from_generation(*defender).genotype.ID

        '''min_attacker_rating = 1000000
        max_attacker_rating = -1000000
        min_defender_rating = 1000000
        max_defender_rating = -1000000'''

        attacker_ranks = dict()
        for objective in self.ratings[attacker_ids[attackers[0]]]:
            attacker_ranks[objective] = list(range(len(attackers)))
            attacker_ranks[objective].sort(key=lambda a: self.ratings[attacker_ids[attackers[a]]][objective],
                                           reverse=True)
            attacker_ranks[objective] = [attacker_ranks[objective].index(i) for i in
                                         range(len(attacker_ranks[objective]))]
        defender_ranks = dict()
        for objective in self.ratings[defender_ids[defenders[0]]]:
            defender_ranks[objective] = list(range(len(defenders)))
            defender_ranks[objective].sort(key=lambda d: self.ratings[defender_ids[defenders[d]]][objective],
                                              reverse=True)
            defender_ranks[objective] = [defender_ranks[objective].index(i) for i in
                                            range(len(defender_ranks[objective]))]

        if not self.use_rank:
            def get_rating_distance(attacker, defender):
                # Direct rating distance
                attacker_ID = self.attacker_generator.get_from_generation(*attacker).genotype.ID
                defender_ID = self.defender_generator.get_from_generation(*defender).genotype.ID
                if (attacker, defender) in self.completed_pairings:
                    return float("inf")
                sum_of_squares = 0
                for objective in self.ratings[attacker_ID]:
                    sum_of_squares += (self.ratings[attacker_ID][objective] - self.ratings[defender_ID][
                        objective]) ** 2
                distance = math.sqrt(sum_of_squares)
                return distance
        else:
            def get_rating_distance(attacker, defender):
                # Rank distance, preventing the failures that occur when the populations have heavily different mean ratings
                attacker_ID = self.attacker_generator.get_from_generation(*attacker).genotype.ID
                defender_ID = self.defender_generator.get_from_generation(*defender).genotype.ID
                if (attacker, defender) in self.completed_pairings:
                    return float("inf")
                sum_of_squares = 0
                for objective in self.ratings[attacker_ID]:
                    sum_of_squares += (attacker_ranks[objective][attackers.index(attacker)] -
                                       defender_ranks[objective][defenders.index(defender)]) ** 2
                distance = math.sqrt(sum_of_squares)
                return distance

        invalid_pairs = list()
        distances = dict()
        for attacker, defender in self.completed_pairings:
            a = attackers.index(attacker)
            d = defenders.index(defender)
            invalid_pairs.append((a, d))

        pairs = self.generate_pairs(len(attackers), len(defenders), 1, invalid_pairs=invalid_pairs)
        # pair_attackers = [attacker for attacker, _ in pairs]
        # pair_defenders = [defender for _, defender in pairs]
        # print(sorted(pair_attackers))
        # print(sorted(pair_defenders))
        opponents = dict()
        for attacker, defender in pairs:
            opponents[attacker] = defender
        distance_sum = sum([distances.setdefault((attacker, defender), get_rating_distance(attackers[attacker], defenders[defender])) for attacker, defender in pairs])
        alternate_distance_sum = sum([distances.setdefault((attacker, defender), get_rating_distance(attackers[attacker], defenders[defender])) for attacker, defender in alternate_pairs])

        print(f"Random distance of {distance_sum}")
        print(f"Greedy distance of {alternate_distance_sum}")
        improvement_steps = 0
        steps_since_improvement = 0
        while steps_since_improvement < 200:
            pair_1, pair_2 = random.choices(opponents, k=2)
            distance_1 = distances.setdefault((pair_1, opponents[pair_1]),
                                              get_rating_distance(attackers[pair_1], defenders[opponents[pair_1]]))
            distance_2 = distances.setdefault((pair_2, opponents[pair_2]),
                                              get_rating_distance(attackers[pair_2], defenders[opponents[pair_2]]))
            original_distance = distance_1 + distance_2
            # new_distance_sum = distance_sum - distance_1 - distance_2
            opponents[pair_1], opponents[pair_2] = opponents[pair_2], opponents[pair_1]
            distance_1 = distances.setdefault((pair_1, opponents[pair_1]),
                                              get_rating_distance(attackers[pair_1], defenders[opponents[pair_1]]))
            distance_2 = distances.setdefault((pair_2, opponents[pair_2]),
                                              get_rating_distance(attackers[pair_2], defenders[opponents[pair_2]]))
            new_distance = distance_1 + distance_2
            # new_distance_sum = new_distance_sum + distance_1 + distance_2
            if new_distance < original_distance:
                # distance_sum = new_distance_sum
                improvement_steps += 1
                steps_since_improvement = 0
            else:
                opponents[pair_1], opponents[pair_2] = opponents[pair_2], opponents[pair_1]
                steps_since_improvement += 1
        # print("Final distance of {0} after {1} improvements".format(distance_sum, improvement_steps))
        print("Hill-climber made {0} improvements.".format(improvement_steps))
        pairs = [(attacker, defender) for attacker, defender in opponents.items()]
        final_distance_sum = sum([distances.setdefault((attacker, defender), get_rating_distance(attackers[attacker], defenders[defender])) for attacker, defender in pairs])
        print(f"Final distance of {final_distance_sum}")
        for attacker, defender in pairs:
            attacker_ID = self.attacker_generator.get_from_generation(*attackers[attacker]).genotype.ID
            defender_ID = self.defender_generator.get_from_generation(*defenders[defender]).genotype.ID
            print(f"{attacker_ID}: {self.ratings[attacker_ID]} <---> {defender_ID}: {self.ratings[defender_ID]}")
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
                for objective in self.ratings[attacker_ID]:
                    sum_of_squares += (self.ratings[attacker_ID][objective] - self.ratings[defender_ID][objective]) ** 2
                distances[-1].append(math.sqrt(sum_of_squares))

        pairs = munkres.Munkres().compute(distances)
        distance_sum = sum([distances[attacker][defender] for attacker, defender in pairs])
        print("Ideal distance of {}".format(distance_sum))
        return pairs
