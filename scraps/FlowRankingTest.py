import random


def generate_pairs(size_1, size_2, pairs_per_individual, invalid_pairs=None):
    if invalid_pairs is None:
        invalid_pairs = {}
    reverse = False
    lesser = size_1
    greater = size_2
    if lesser > greater:
        lesser, greater = greater, lesser
        reverse = True
    if pairs_per_individual > lesser:
        print("Warning: low population size will lead to repeat evaluations.")

    pairs = list()
    lesser_count = dict()

    for i in range(lesser):
        lesser_count[i] = 0

    for j in range(greater):
        valid = [i for i in range(lesser) if (i, j) not in pairs and (i, j) not in invalid_pairs]
        random.shuffle(valid)
        valid.sort(key=lesser_count.get)
        for p in range(pairs_per_individual):
            i = valid[p % len(valid)]  # Modulus in case of lower population than evaluations per individual
            pairs.append((i, j))
            lesser_count[i] += 1
    if reverse:
        pairs = [(j, i) for (i, j) in pairs]
    return pairs


def flow_ranking(attacker_wins, defender_wins):
    attacker_size = len(attacker_wins)
    defender_size = len(defender_wins)
    attacker_losses = {attacker: {defender for defender in defender_wins if attacker in defender_wins[defender]}
                       for attacker in attacker_wins}
    defender_losses = {defender: {attacker for attacker in attacker_wins if defender in attacker_wins[attacker]}
                       for defender in defender_wins}
    attacker_points = [100 for _ in range(attacker_size)]
    defender_points = [100 for _ in range(defender_size)]

    for _ in range(100):
        recycled_points = sum([attacker_points[attacker] for attacker in attacker_losses if len(attacker_losses[attacker]) == 0])
        recycled_points += sum([defender_points[defender] for defender in defender_losses if len(defender_losses[defender]) == 0])
        recycled_points /= (attacker_size + defender_size)
        new_attacker_points = [100 + recycled_points for _ in range(attacker_size)]
        new_defender_points = [100 + recycled_points for _ in range(defender_size)]
        for attacker in range(attacker_size):
            for defender in attacker_wins[attacker]:
                new_attacker_points[attacker] += defender_points[defender]
        for defender in range(defender_size):
            for attacker in defender_wins[defender]:
                new_defender_points[defender] += attacker_points[attacker]
        attacker_points = [points - 100 for points in new_attacker_points]
        defender_points = [points - 100 for points in new_defender_points]
    return attacker_points, defender_points


def page_ranking(attacker_wins, defender_wins):
    damping_factor = 1.0
    attacker_size = len(attacker_wins)
    defender_size = len(defender_wins)
    total_size = attacker_size + defender_size
    attacker_losses = {attacker: {defender for defender in defender_wins if attacker in defender_wins[defender]}
                       for attacker in attacker_wins}
    defender_losses = {defender: {attacker for attacker in attacker_wins if defender in attacker_wins[attacker]}
                       for defender in defender_wins}
    attacker_points = [1 for _ in range(attacker_size)]
    defender_points = [1 for _ in range(defender_size)]
    damping_points = (1 - damping_factor)

    for _ in range(100):
        new_attacker_points = list()
        new_defender_points = list()
        sink_pool = 0
        for attacker in range(attacker_size):
            transfer = 0
            for defender in attacker_wins[attacker]:
                transfer += defender_points[defender] / len(defender_losses[defender])
            new_attacker_points.append(damping_points + damping_factor * transfer)
            if len(attacker_losses[attacker]) == 0:
                sink_pool += attacker_points[attacker] / total_size
        for defender in range(defender_size):
            transfer = 0
            for attacker in defender_wins[defender]:
                transfer += attacker_points[attacker] / len(attacker_losses[attacker])
            new_defender_points.append(damping_points + damping_factor * transfer)
            if len(defender_losses[defender]) == 0:
                sink_pool += defender_points[defender] / total_size

        attacker_points = [points + sink_pool for points in new_attacker_points]
        defender_points = [points + sink_pool for points in new_defender_points]
    return attacker_points, defender_points



if __name__ == "__main__":
    attackers = [random.random() for _ in range(10)]
    defenders = [random.random() for _ in range(10)]
    attackers.sort(reverse=True)
    defenders.sort(reverse=True)
    pairs = generate_pairs(10, 10, 3)
    attacker_wins = {i: set() for i in range(len(attackers))}
    defender_wins = {i: set() for i in range(len(defenders))}
    for attacker, defender in pairs:
        attacker_score = attackers[attacker]#random.gauss(attackers[attacker], 0.25)
        defender_score = defenders[defender]#random.gauss(defenders[defender], 0.25)
        if attacker_score > defender_score:
            attacker_wins[attacker].add(defender)
        else:
            defender_wins[defender].add(attacker)

    #attacker_wins = {0: {0, 1}, 1: {}}
    #defender_wins = {0: {1}, 1: {1}}

    attacker_ranks, defender_ranks = page_ranking(attacker_wins, defender_wins)
    pass