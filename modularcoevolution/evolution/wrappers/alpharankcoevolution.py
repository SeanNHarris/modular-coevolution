from modularcoevolution.evolution.wrappers.similarstrengthcoevolution import SimilarStrengthCoevolution

from open_spiel.python.egt import alpharank

import numpy

import math
import faulthandler

class AlphaRankCoevolution(SimilarStrengthCoevolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        faulthandler.enable()

    def normalize_objective(self, objective, value):
        return value / (self.max_objectives[objective] - self.min_objectives[objective])

    def calculate_ratings(self, ratings_to_update):
        current_attacker_ids = [self.attacker_generator.get_from_generation(*attacker_index).genotype.id
                                for attacker_index in self.current_attackers]
        current_defender_ids = [self.defender_generator.get_from_generation(*defender_index).genotype.id
                                for defender_index in self.current_defenders]
        num_agents = len(current_attacker_ids) + len(current_defender_ids)
        example_attacker, example_defender = self.get_pair(self.scored_pairings[0])
        for objective in self.scores_per_opponent[example_attacker.genotype.id][example_defender.genotype.id]:
            payoff_matrix = numpy.zeros((2, num_agents, num_agents))
            for evaluation_id in self.scored_pairings:
                attacker, defender = self.get_pair(evaluation_id)
                attacker_id = attacker.genotype.id
                defender_id = defender.genotype.id
                assert attacker_id in current_attacker_ids
                assert defender_id in current_defender_ids
                attacker_scores = self.scores_per_opponent[attacker_id][defender_id][objective]
                attacker_payoff = self.normalize_objective(objective, sum(attacker_scores) / len(attacker_scores))
                defender_scores = self.scores_per_opponent[defender_id][attacker_id][objective]
                defender_payoff = self.normalize_objective(objective, sum(defender_scores) / len(defender_scores))

                #assert attacker_payoff == -defender_payoff

                attacker_index = current_attacker_ids.index(attacker_id)
                defender_index = current_defender_ids.index(defender_id) + len(current_attacker_ids)
                # assert attacker_index != defender_index
                payoff_matrix[0, attacker_index, defender_index] = attacker_payoff
                payoff_matrix[1, attacker_index, defender_index] = defender_payoff
                payoff_matrix[0, defender_index, attacker_index] = defender_payoff
                payoff_matrix[1, defender_index, attacker_index] = attacker_payoff

            # assert numpy.array_equal(payoff_matrix[0], payoff_matrix[1].T)
            # heuristic_payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_matrix[0])]
            # heuristic_payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_matrix[0]),
            #                            heuristic_payoff_table.from_matrix_game(payoff_matrix[1].T)]
            # assert numpy.array_equal(heuristic_payoff_tables[0](), heuristic_payoff_tables[1]())
            # Simplifies heuristic_payoff_tables if the game is symmetric
            # is_symmetric, heuristic_payoff_tables = utils.is_symmetric_matrix_game(heuristic_payoff_tables)
            #assert is_symmetric

            (rhos, rho_m, pi, num_profiles, num_strats_per_population) =\
                alpharank.compute(payoff_matrix[0:1], alpha=1e2)
            # alpharank.compute_and_report_alpharank(heuristic_payoff_tables, alpha=1e2)
            print(f"Pi: {sorted(pi, reverse=True)}")
            for index, attacker_id in enumerate(current_attacker_ids):
                if pi[index] > 0:
                    self.ratings[attacker_id][objective] = math.log(pi[index])
                else:
                    self.ratings[attacker_id][objective] = -100
            for index, defender_id in enumerate(current_defender_ids):
                if pi[index + len(current_attacker_ids)] > 0:
                    self.ratings[defender_id][objective] = math.log(pi[index + len(current_attacker_ids)])
                else:
                    self.ratings[defender_id][objective] = -100

        '''
        # Calculate connected components
        all_ids = set()
        components = list()
        all_ids.update(current_defender_ids)
        all_ids.update(current_attacker_ids)
        found_ids = set()
        while len(found_ids) < len(all_ids):
            frontier = set()
            frontier.add((all_ids - found_ids).pop())
            connected = set()
            while len(frontier) > 0:
                current = frontier.pop()
                found_ids.add(current)
                connected.add(current)
                for opponent in self.scores_per_opponent[current]:
                    if opponent not in found_ids:
                        frontier.add(opponent)
            components.append(connected)
        print(components)
        '''
