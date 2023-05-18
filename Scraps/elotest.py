import math
import random


# Expected score for player 1
def expected_score(rating_1, rating_2):
    return 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating_2 - rating_1) / 400))


def update_score(rating_1, rating_2, score_1, k_factor=40):
    score_2 = 1 - score_1
    expected_score_1 = expected_score(rating_1, rating_2)
    expected_score_2 = 1 - expected_score_1
    rating_1_update = rating_1 + k_factor * (score_1 - expected_score_1)
    rating_2_update = rating_2 + k_factor * (score_2 - expected_score_2)
    return rating_1_update, rating_2_update, score_1 - expected_score_1


def get_pairings(attacker_ratings, defender_ratings):
    attacker_rating_order = sorted(range(len(attacker_ratings)), key=attacker_ratings.__getitem__)
    defender_rating_order = sorted(range(len(defender_ratings)), key=defender_ratings.__getitem__)
    return [(attacker, defender) for attacker, defender in zip(attacker_rating_order, defender_rating_order)]


def get_pairings_selective(attacker_ratings, defender_ratings, previous_pairings=None):
    if previous_pairings is None:
        previous_pairings = {}
    pairings = set()
    for attacker, attacker_rating in enumerate(attacker_ratings):
        closest_opponent = random.randint(0, len(defender_ratings) - 1)
        closest_opponent_distance = abs(attacker_rating - defender_ratings[closest_opponent])
        for defender, defender_rating in enumerate(defender_ratings):
            if (attacker, defender) in previous_pairings:
                continue
            rating_distance = abs(attacker_rating - defender_rating)
            if closest_opponent is None or rating_distance < closest_opponent_distance:
                closest_opponent = defender
                closest_opponent_distance = rating_distance
        assert closest_opponent is not None
        pairings.add((attacker, closest_opponent))
    for defender, defender_rating in enumerate(defender_ratings):
        closest_opponent = random.randint(0, len(attacker_ratings) - 1)
        closest_opponent_distance = abs(attacker_ratings[closest_opponent] - defender_rating)
        for attacker, attacker_rating in enumerate(attacker_ratings):
            if (attacker, defender) in previous_pairings:
                continue
            rating_distance = abs(defender_rating - attacker_rating)
            if closest_opponent is None or rating_distance < closest_opponent_distance:
                closest_opponent = attacker
                closest_opponent_distance = rating_distance
        assert closest_opponent is not None
        pairings.add((closest_opponent, defender))
    return list(pairings)
        


if __name__ == "__main__":
    results = list()
    for _ in range(10000):
        # attackers = [random.random() for _ in range(10)]
        # defenders = [random.random() for _ in range(10)]
        attackers = [0.1 * i for i in range(1, 11)]
        defenders = [0.1 * i for i in range(1, 11)]
        attackers.sort(reverse=True)
        defenders.sort(reverse=True)
        attacker_ratings = [0 for attacker in attackers]
        defender_ratings = [0 for defender in defenders]
        max_score = 1
        min_score = -1
        pairings = list()
        previous_pairings = list()
        score_deviance_window = list()
        average_score_deviance = 1
        game_count = 0
        while game_count < 4*10:# or abs(average_score_deviance) > 0.001:
            if len(pairings) == 0:
                #pairing_map = get_pairings(attacker_ratings, defender_ratings)
                pairings = get_pairings_selective(attacker_ratings, defender_ratings, previous_pairings)
                previous_pairings.extend(pairings.copy())
            if random.random() < 1.0:
                attacker, defender = pairings.pop(0)
            else:
                attacker = random.randint(0, len(attackers) - 1)
                defender = random.randint(0, len(defenders) - 1)
            attacker_score = random.gauss(attackers[attacker], 0.0)
            defender_score = random.gauss(defenders[defender], 0.0)
            attacker_relative_score = attacker_score - defender_score
            if attacker_relative_score > max_score:
                max_score = attacker_relative_score
            if attacker_relative_score < min_score:
                min_score = attacker_relative_score
            attacker_relative_score -= min_score
            attacker_relative_score /= (max_score - min_score)
            # print("Match between {} ({}) and {} ({}): attacker score of {}".format(attacker, attackers[attacker], defender, defenders[defender], attacker_relative_score))
            attacker_ratings[attacker], defender_ratings[defender], score_deviance = \
                update_score(attacker_ratings[attacker], defender_ratings[defender], attacker_relative_score)
            score_deviance_window.append(score_deviance)
            if len(score_deviance_window) > 100:
                score_deviance_window.pop(0)
            average_score_deviance = sum(score_deviance_window) / len(score_deviance_window)
            # print("Average score deviance: {}".format(average_score_deviance))
            game_count += 1
        print("Convergence reached after {} games".format(game_count))
        print(attacker_ratings)
        print(defender_ratings)
        results.append(game_count)
    print("Average number of games for convergence: {}".format(sum(results) / len(results)))
