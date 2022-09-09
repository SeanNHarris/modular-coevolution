import random


# Edit diversity from diversity in Genetic Programming: An Analysis of Measures and Correlation With Fitness
def edit_distance(tree_1, tree_2):
    return subtree_edit_distance(tree_1.root, tree_2.root)


def subtree_edit_distance(node_1, node_2):
    distance = 0
    if node_1 is None or node_2 is None:
        distance += 1
    elif node_1.function_id != node_2.function_id or node_1.literal != node_2.literal:
        distance += 1
    node_1_inputs = 0
    if node_1 is not None:
        node_1_inputs = len(node_1.input_nodes)
    node_2_inputs = 0
    if node_2 is not None:
        node_2_inputs = len(node_2.input_nodes)
    if node_1_inputs == 0 and node_2_inputs == 0:
        return distance

    for i in range(max(node_1_inputs, node_2_inputs)):
        subtree_1 = None
        if i < node_1_inputs:
            subtree_1 = node_1.input_nodes[i]
        subtree_2 = None
        if i < node_2_inputs:
            subtree_2 = node_2.input_nodes[i]
        distance += 0.5 * subtree_edit_distance(subtree_1, subtree_2)
    return distance


def edit_diversity(population, reference=None, samples=0):
    if reference is None and any(not i.fitness for i in population):
        reference = population[0]
    if reference is None:
        reference = max(population, key=lambda x: x.fitness)   # Note: diversity score is in reference to the best individual?
    distance_sum = 0
    if samples is None:
        comparisons = population
    else:
        comparisons = random.sample(population, samples)
    for comparison in comparisons:
        distance_sum += edit_distance(reference, comparison)
    return distance_sum / samples
