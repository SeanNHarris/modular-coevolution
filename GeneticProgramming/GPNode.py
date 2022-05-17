# uncompyle6 version 2.13.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: S:\My Documents\ceads\ceads\CEADS-LIN\GeneticProgramming\GPNodes.py
# Compiled at: 2017-03-21 14:16:04
# Size of source mod 2**32: 7822 bytes
from GeneticProgramming.GPNodeTypeRegistry import GPNodeTypeRegistry

import abc
import random


class GPNode(metaclass=GPNodeTypeRegistry):
    functions = None
    literals = None

    type_functions = None
    terminal_list = None
    non_terminal_list = None
    branch_list = None
    semiterminal_table = None

    DATA_TYPES = None

    def __init__(self, output_type=None, function_id=-1, literal=None, terminal=False, non_semi_terminal=False,
                 non_terminal=False, branch=False, forbidden_nodes=None, fixed_context=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        if function_id == -1:
            function_id = type(self).random_function(output_type, terminal, non_semi_terminal, non_terminal, branch,
                                                     forbidden_nodes=forbidden_nodes)
            if function_id is None:
                raise Exception("Impossible constraints on GP node selection!")
        self.function_id = function_id
        func_data = type(self).get_function(function_id)
        self.function = func_data[0]
        self.output_type = func_data[1]
        self.input_types = func_data[2]
        self.input_nodes = list()
        self.literal = literal
        self.fixed_context = fixed_context
        if function_id in type(self).literals and literal is None:
            self.literal = self.function(fixed_context)
        self.parent = None

    def execute(self, context):
        output = None
        if self.function_id not in type(self).literals:
            output = self.function(self.input_nodes, context)
        else:
            output = self.literal
        return output

    def add_input(self, input_node):
        self.input_nodes.append(input_node)

    def set_parent(self, parent):
        self.parent = parent

    def get_height(self):
        if len(self.input_nodes) == 0:
            return 1
        else:
            max_length = 0
            for node in self.input_nodes:
                max_length = max(max_length, node.get_height())

            return max_length + 1

    def get_size(self):
        return sum((1 for _ in self.traverse_post_order()))

    def get_node_list(self):
        node_list = [self]
        for input_node in self.input_nodes:
            node_list.extend(input_node.get_node_list())

        return node_list

    def tree_string(self, max_height, depth):
        string = ''
        for _ in range(depth):
            string += '|\t'

        string += str(self)
        string += '\n'
        for input_node in self.input_nodes:
            string += input_node.tree_string(max_height, depth + 1)

        return string

    def traverse_post_order(self):
        for child in self.input_nodes:
            for node in child.traverse_post_order():
                yield node

        yield self

    def traverse_pre_order(self):
        yield self
        for child in self.input_nodes:
            for node in child.traverse_pre_order():
                yield node

    def __str__(self):
        if self.function_id not in type(self).literals:
            return self.function.__name__
        else:
            return self.function.__name__ + '(' + str(self.literal) + ')'

    @classmethod
    def build_data_type_tables(cls):
        cls.type_functions = dict()
        for data_type in cls.DATA_TYPES:
            cls.type_functions[data_type] = list()

        cls.terminal_list = list()
        cls.non_terminal_list = list()
        cls.branch_list = list()
        for node in range(len(cls.functions)):
            node_data = cls.get_function(node)
            cls.type_functions[node_data[1]].append(node)
            if len(node_data[2]) == 0:
                cls.terminal_list.append(node)
            else:
                cls.non_terminal_list.append(node)
                if len(node_data[2]) > 1:
                    cls.branch_list.append(node)

        if cls.functions is None:
            cls.functions = list()
        if cls.literals is None:
            cls.literals = list()

    @classmethod
    def build_height_tables(cls, height, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        grow_table = dict()
        full_table = dict()
        grow_table[1] = set()
        full_table[1] = set()
        for function in cls.terminal_list:
            if function in forbidden_nodes:
                continue
            grow_table[1].add(cls.get_function(function)[1])
            full_table[1].add(cls.get_function(function)[1])

        for i in range(2, height + 1):
            grow_table[i] = set()
            full_table[i] = set()
            grow_table[i] |= grow_table[i - 1]
            for function in cls.non_terminal_list:
                if function in forbidden_nodes:
                    continue
                grow_valid = True
                full_valid = True
                for inputType in cls.get_function(function)[2]:
                    if inputType not in grow_table[i - 1]:
                        grow_valid = False
                    if inputType not in full_table[i - 1]:
                        full_valid = False

                if grow_valid:
                    grow_table[i].add(cls.get_function(function)[1])
                if full_valid:
                    full_table[i].add(cls.get_function(function)[1])

        return grow_table, full_table

    @classmethod
    def get_semi_terminals(cls):
        if cls.semiterminalList is None:
            cls.semiterminalList = list()
            cls.semiterminalList.extend(cls.terminal_list)
            dead_types = list()
            while 1:
                live_functions = [function for function in cls.non_terminal_list if
                                  function not in cls.semiterminalList]
                for data_type in cls.DATA_TYPES:
                    if len(set(live_functions) & set(cls.type_functions[data_type])) == 0:
                        dead_types.append(data_type)

                modify = False
                for function in live_functions:
                    still_alive = False
                    for inputType in cls.get_function(function)[2]:
                        if inputType not in dead_types:
                            still_alive = True

                    if not still_alive:
                        cls.semiterminalList.append(function)
                        modify = True

                if not modify:
                    break

            print(cls.semiterminalList)
        return cls.semiterminalList

    @classmethod
    def build_semiterminal_table(cls, height, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        if cls.semiterminal_table is None:
            cls.semiterminal_table = dict()
            cls.semiterminal_table[1] = set(
                [function for function in cls.terminal_list if function not in forbidden_nodes])
            for i in range(2, height + 1):
                cls.semiterminal_table[i] = set()
            dead_types = set()
            for i in range(2, height + 1):
                cls.semiterminal_table[i] = set()
                live_functions = [function for function in cls.non_terminal_list if
                                  function not in cls.semiterminal_table[i - 1] and function not in forbidden_nodes]
                for dataType in cls.DATA_TYPES:
                    if len(set(live_functions) & set(cls.type_functions[dataType])) == 0:
                        dead_types.add(dataType)

                modify = False
                for function in live_functions:
                    still_alive = False
                    for inputType in cls.get_function(function)[2]:
                        if inputType not in dead_types:
                            still_alive = True

                    if not still_alive:
                        cls.semiterminal_table[i].add(function)
                        modify = True

                if not modify:
                    break

            print(cls.semiterminal_table)
        return cls.semiterminal_table

    # Generates a table describing how deep it's possible to construct a tree with a given type.
    @classmethod
    def build_depth_table(cls, max_height, forbidden_nodes=None):
        # It's clearly impossible to go more than this depth without hitting a dead end or loop
        # maxHeight = len(cls.DATA_TYPES)

        if forbidden_nodes is None:
            forbidden_nodes = []
        type_tables = dict()
        type_depths = dict()
        for type in cls.DATA_TYPES:
            type_tables[type] = {1: {type}}
            type_depths[type] = 1
        # Construct the type tables describing which types can be present at each depth by taking the list of children of possible types at the previous depth
        for depth in range(2, max_height + 1):
            for type in cls.DATA_TYPES:
                children_types = set()
                for parent_type in type_tables[type][depth - 1]:
                    edge_functions = [function for function in range(len(cls.functions)) if
                                      function in cls.type_functions[parent_type] and function not in forbidden_nodes]
                    for function in edge_functions:
                        children_types |= set(cls.get_function(function)[2])
                type_tables[type][depth] = children_types
        # Construct the depth table, listing the highest depth that can be reached from a type.
        for depth in range(2, max_height + 1):
            for type in cls.DATA_TYPES:
                # Loops indicate that infinite depth can be reached
                if type in type_tables[type][depth]:
                    type_depths[type] = 1000000
                    continue
                # If anything can be reached at the current depth, then that depth can be reached
                if len(type_tables[type][depth]) > 0:
                    type_depths[type] = max(type_depths[type], depth)
                # If a type can be reached is known to have an even higher max depth (usually due to a loop) then that higher depth can be reached
                for childType in type_tables[type][depth]:
                    type_depths[type] = max(type_depths[type], type_depths[childType])
        return type_depths

    @classmethod
    def random_function(cls, output_type, terminal=False, non_semi_terminal=False, non_terminal=False, branch=False,
                        num_children=None, child_types=None, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = []
        possible = list([function for function in cls.type_functions[output_type] if function not in forbidden_nodes])
        if terminal:
            new_possible = [function for function in possible if function in cls.terminal_list]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        if non_semi_terminal:
            new_possible = [function for function in possible if function not in cls.get_semi_terminals()]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        if non_terminal:
            new_possible = [function for function in possible if function in cls.non_terminal_list]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        if branch:
            new_possible = [function for function in possible if function in cls.branch_list]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        if num_children is not None:
            new_possible = [function for function in possible if len(cls.get_function(function)[2]) == num_children]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        if child_types is not None:
            new_possible = [function for function in possible if cls.get_function(function)[2] == child_types]
            if len(new_possible) > 0:
                possible = new_possible
            else:
                return None
        return random.choice(possible)

    @classmethod
    def get_function(cls, func_id):
        return cls.functions[func_id]

    @classmethod
    def get_id(cls, function):
        return [func_id for func_id, func_tuple in enumerate(cls.functions) if func_tuple[0] == function][0]

    @classmethod
    def gp_primitive(cls, output_type, input_types):
        def internal_decorator(function):
            if cls.functions is None:
                cls.functions = list()
            cls.functions.append((function, output_type, input_types))
            return function
        return internal_decorator

    @classmethod
    def gp_literal(cls, output_type):
        def internal_decorator(function):
            if cls.functions is None:
                cls.functions = list()
            if cls.literals is None:
                cls.literals = list()
            function_id = len(cls.functions)
            cls.functions.append((function, output_type, ()))
            cls.literals.append(function_id)
            return function
        return internal_decorator
