# Represents a genetic programming tree. Nodes for the tree are in GPNodes.py.
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.geneticprogramming.gpnode import GPNodeTypeRegistry
from modularcoevolution.genotypes.diversity.gpdiversity import *

from typing import Any, TypedDict, Union

import random

# BRANCH_CHANCE = 0.5
MAXIMUM_HEIGHT = 100  # Not enforced, used for precalculation of type tables; trees larger than this will cause a crash

# TODO: Rename variables


class GPTreeParameters(TypedDict, total=False):
    node_type: Union[type, str]
    return_type: int  # GPNode type
    min_height: int
    max_height: int
    parsimony_weight: float
    forbidden_nodes: list[int]  # list of GPNode IDs
    fixed_context: list[str, Any]  # Also specific, can we do more typing enforcement with this?
    id_list: list[int]  # list of GPNode IDs


class GPTree(BaseGenotype):
    # Generate a tree either at random or from a list of expansions.
    def __init__(self, parameters):
        super().__init__()

        self.parameters = parameters
        if "node_type" in parameters:
            if isinstance(parameters["node_type"], type):
                self.nodeType = parameters["node_type"]
            elif isinstance(parameters["node_type"], str):
                self.nodeType = GPNodeTypeRegistry.name_lookup[parameters["node_type"]]
            else:
                raise TypeError("nodeType must be a type or a type name.")
        else:
            raise TypeError("A node type must be supplied as a parameter.")

        if "return_type" in parameters:
            self.returnType = parameters["return_type"]
        else:
            self.returnType = 0  # Should be void

        if "min_height" in parameters:
            self.minHeight = parameters["min_height"]
        else:
            self.minHeight = 3

        if "max_height" in parameters:
            self.maxHeight = parameters["max_height"]
        else:
            self.maxHeight = 7

        # parsimonyWeight - a percentage of fitness removed per node in the tree, should be on the order of 0.01 = 1%
        if "parsimony_weight" in parameters:
            self.parsimony_weight = parameters["parsimony_weight"]
        else:
            self.parsimony_weight = 0.0025

        if "forbidden_nodes" in parameters:
            self.forbiddenNodes = parameters["forbidden_nodes"]
        else:
            self.forbiddenNodes = list()

        if "fixed_context" in parameters:
            self.fixed_context = parameters["fixed_context"]
        else:
            self.fixed_context = dict()

        self.nodeType.build_data_type_tables()
        self.growTable, self.fullTable = self.nodeType.build_height_tables(MAXIMUM_HEIGHT,
                                                                           self.forbiddenNodes)  # Todo: Cache these per node type, forbiddenNodes
        self.depthTable = self.nodeType.build_depth_table(MAXIMUM_HEIGHT, self.forbiddenNodes)

        if "id_list" in parameters:
            self.root = self.generateFromList(list(parameters["id_list"]))
        else:
            height = random.randint(self.minHeight, self.maxHeight)
            self.root = self.randomSubtree(height, self.returnType)

    # print("Tree generated:")
    # print(self)

    # Run the program the tree represents
    def execute(self, context):
        # print("Executing tree:\n" + str(self))
        context = context.copy()
        context.update(self.fixed_context)
        return self.root.execute(context)

    # Replaces a subtree with a replacement subtree
    def replace(self, node, replacement):
        # print("Replacing node:")
        # print(node.treeString(node.getHeight(), 0))
        # print("With:")
        # print(replacement.treeString(node.getHeight(), 0))

        if node is self.root:
            self.root = replacement
            if self.root.get_height() > MAXIMUM_HEIGHT:
                self.root = node
        else:
            parent = node.parent
            index = parent.input_nodes.index(node)
            parent.input_nodes[index] = replacement
            replacement.set_parent(parent)
            if self.root.get_height() > MAXIMUM_HEIGHT:
                parent.input_nodes[index] = node

    # Selects a random node from the tree
    def randomNode(self, outputType=None):
        return random.choice(self.root.get_node_list())

    def mutate(self):
        mutation_function = random.choices((self.subtree_mutate, self.point_mutate), (0.5, 0.5), k=1)[0]
        mutation_function()

    def subtree_mutate(self):
        oldSubtree = self.randomNode()
        meanHeight = oldSubtree.get_height()
        realHeight = int(round(random.gauss(meanHeight, 1)))
        newSubtree = self.randomSubtree(realHeight, oldSubtree.output_type)
        self.replace(oldSubtree, newSubtree)
        try:
            clone = self.clone()
        except:
            print("Mutation produced invalid tree!")
            raise Exception
        self.creation_method = "Mutation"

    def point_mutate(self, mutation_amount = None):
        if mutation_amount is None:
            mutation_amount = 1 / len(self)
        for node in self.root.get_node_list():
            if random.random() < mutation_amount:
                new_node = self.nodeType(function_id=self.nodeType.random_function(node.output_type, child_types=node.input_types), fixed_context=self.fixed_context)
                if node is self.root:
                    self.root = new_node
                else:
                    new_node.set_parent(node.parent)
                    node.parent.input_nodes[node.parent.input_nodes.index(node)] = new_node
                for child in node.input_nodes:
                    new_node.add_input(child)
                    child.set_parent(new_node)
        try:
            clone = self.clone()
        except:
            print("Mutation produced invalid tree!")
            raise Exception
        self.creation_method = "Mutation"

    # Subtree recombination
    def recombine(self, donor):
        donorSubtree = donor.randomNode()
        nodeList = [node for node in self.root.get_node_list() if node.output_type == donorSubtree.output_type]
        if len(nodeList) > 0:
            self.replace(random.choice(nodeList), donorSubtree)
        try:
            clone = self.clone()
        except:
            print("Recombination produced invalid tree!")
            raise Exception
        self.parent_ids.append(donor.id)
        self.creation_method = "Recombination"

    # Creates a deep copy of the tree
    def clone(self, copy_objectives=False):
        cloned_genotype = GPTree({**self.parameters, "id_list": self.getNodeIDList()})
        if copy_objectives:
            for objective in self.objectives:
                cloned_genotype.objectives[objective] = self.objectives[objective]
                cloned_genotype.objective_statistics[objective] = self.objective_statistics[objective]
                cloned_genotype.objectives_counter[objective] = self.objectives_counter[objective]
                cloned_genotype.past_objectives[objective] = self.past_objectives[objective]
            cloned_genotype.evaluated = True
            cloned_genotype.fitness = self.fitness
        cloned_genotype.parent_ids.append(self.id)
        cloned_genotype.creation_method = "Cloning"
        return cloned_genotype

    # Computes an id list for the tree, which is just a preorder list of node ids.
    def getNodeIDList(self):
        idList = list()
        for node in self.root.traverse_pre_order():
            idList.append(node.function_id)
            if node.function_id in self.nodeType.literals:
                idList.append(node.literal)
        return idList

    def __str__(self):
        treeString = self.root.tree_string(self.root.get_height(), 0)
        return treeString + "ID List: " + str(self.getNodeIDList()) + "\n"

    def __len__(self):
        return len(self.root.get_node_list())

    # Positive is good, negative is bad
    def get_fitness_modifier(self, raw_fitness):
        assert self.parsimony_weight > 0  # Check that the old parameter is not being used.
        parsimony_pressure = min(0.9, self.parsimony_weight * len(self))  # Cap penalty at 90%
        fitness_modifier = -parsimony_pressure * abs(raw_fitness)  # Percentage penalty based on size
        # print("Parsimony pressure penalty of " + str(parsimony_pressure))
        return fitness_modifier

    # def getTransmitString(self):
    #    idList = self.getNodeIDList()
    #    idString = pickle.dumps(idList)
    #    return idString

    # Generate a tree given an id list
    def generateFromList(self, idList):
        function = idList.pop(0)
        literal = None
        if function in self.nodeType.literals:
            literal = idList.pop(0)

        node = self.nodeType(function_id=function, literal=literal, fixed_context=self.fixed_context)
        for inputType in node.input_types:
            child = self.generateFromList(idList)
            child.set_parent(node)
            node.add_input(child)
        return node

    # Generate a random tree with the requested height and output type
    def randomSubtree(self, height, outputType):
        subtree = self.generateDeep(height, outputType)
        return subtree

    # Generates a tree guaranteed to be a certain depth
    def generateDeep(self, height, outputType):
        node = None
        if height > 1:
            # Find the list of possible functions to choose that can reach the desired depth
            typeChildren = self.nodeType.type_functions[outputType]
            deepChildren = list()
            for function in typeChildren:
                if function in self.forbiddenNodes:
                    continue
                functionInfo = self.nodeType.get_function(function)
                # Such functions have an input of a type that can reach exactly one less than that depth
                if self.growTable[height - 1].issuperset(functionInfo[2]):  # Ensure that the function can stop at that depth (Using the depth table)
                    for inputType in functionInfo[2]:
                        if self.depthTable[inputType] >= height - 1:  # Ensure that the function can reach that depth (Using the Grow restrictions for typed GP)
                            deepChildren.append(function)
                            break

            # This can only happen if it is literally impossible to build a tree of the desired height. Relax height limit. #Todo: Incorporate more detailed checks into depth table
            if len(deepChildren) == 0:
                print("Failure in full table (type 1). Relaxing restrictions.")
                return self.generateGrow(height, outputType)

            # Select a random function
            node = self.nodeType(outputType, random.choice(deepChildren), fixed_context=self.fixed_context)

            # Assign the inputs with a random start, picking one to generate to the desired depth and generating the rest through grow
            randStart = 0
            if (len(node.input_types) > 0):
                randStart = random.randrange(len(node.input_types))
            deepComplete = False
            inputs = [None] * len(node.input_types)
            for i in range(randStart, randStart + len(node.input_types)):
                inputType = node.input_types[i % len(node.input_types)]
                child = None
                if not deepComplete and self.depthTable[inputType] >= height - 1:
                    child = self.generateDeep(height - 1, inputType)
                    deepComplete = True
                else:
                    child = self.generateGrow(height - 1, inputType)
                inputs[i % len(node.input_types)] = child

            for child in inputs:
                child.set_parent(node)
                node.add_input(child)

        # Use terminal nodes at the depth limit
        else:
            if self.nodeType.random_function(outputType, terminal=True, forbidden_nodes=self.forbiddenNodes) is None:
                print("Failure in full table (type 2). Relaxing restrictions.")
                return self.generateGrow(height, outputType)
            node = self.nodeType(outputType, terminal=True, forbidden_nodes=self.forbiddenNodes, fixed_context=self.fixed_context)

        return node

    # Generates a tree using the grow method for strongly-typed GP
    def generateGrow(self, height, outputType):
        node = None
        if height > 1:
            typeList = self.nodeType.type_functions[outputType]
            randStart = random.randrange(len(typeList))
            for i in range(randStart, randStart + len(typeList)):
                function = typeList[i % len(typeList)]
                if function in self.forbiddenNodes:
                    continue
                functionInfo = self.nodeType.get_function(function)
                if self.growTable[height - 1].issuperset(functionInfo[2]):
                    node = self.nodeType(outputType, function, fixed_context=self.fixed_context)
                    break
            if node is None:
                print("Failure in grow table (type 1). Relaxing restrictions.")
                return self.generateGrow(height + 1, outputType)

        else:
            if self.nodeType.random_function(outputType, terminal=True, forbidden_nodes=self.forbiddenNodes) is None:
                print("Failure in grow table (type 2). Relaxing restrictions.")
                return self.generateGrow(height + 1, outputType)
            node = self.nodeType(outputType, terminal=True, forbidden_nodes=self.forbiddenNodes, fixed_context=self.fixed_context)

        for inputType in node.input_types:
            child = self.generateGrow(height - 1, inputType)
            child.set_parent(node)
            node.add_input(child)
        return node

    def get_raw_genotype(self):
        return {"node_type": str(self.nodeType), "id_list": self.getNodeIDList()}

    def __hash__(self):
        return tuple(self.getNodeIDList()).__hash__()

    def diversity_function(self, population, reference=None, samples=None):
        return edit_diversity(population, reference, samples)