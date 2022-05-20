# GeneticProgramming
This subdirectory contains the `GPTree` class used by all genetic programming trees, and the base class for `GPNode`s.

- `GPNode` is an abstract base class of all genetic programming node grammars, and also contains the general behavior of a GP node.
    The `execute` method takes in a `context` dictionary to be referenced by certain nodes, executes the whole subtree from this node, and returns the tree's return value.
    These nodes use strongly typed genetic programming, and thus a type system must be provided in addition to a list of functions.
    
    Functions to be used as GP nodes must take two parameters, `input_nodes` (a list of child nodes), and `context` (provided by `execute`).
    Child nodes should be called using that child's `execute(context)` method.
    To register a function as a GP node, the `gp_primitive(output_type, input_types)` function decorator should be applied, where `output_type` is the return type of the node, and `input_types` is a tuple specifying the types of that node's children.
    Note that this decorator is a class method.
    
    Consider a class `VectorMathGPNode` with types `REAL` and `VECTOR`. An example `dot_product` node would be created as follows:
    ```python
    @VectorMathGPNode.gp_primitive(REAL, (VECTOR, VECTOR))
    def dot_product(input_nodes, context):
        vector_1 = input_nodes[0].execute(context)
        vector_2 = input_nodes[1].execute(context)
        return sum([value_1 * value_2 for value_1, value_2 in zip(vector_1, vector_2)])
    ```
    
    To create a literal node (i.e. a node that just stores and returns a specific value) for a given data type, create a function that returns a random value to assign to that literal, and register it using the `gp_literal(output_type)` function decorator, as with normal GP primitives.
    The assigned value will not change after it is generated.
    
    These will be used to construct a grammar where the nth input to a node requesting a certain type will be another node returning that type.
    Additionally, the following class variable must be set:
    - `DATA_TYPES` - A list of valid types for GP nodes, matching those used in function registration. (it is recommended to name these, e.g. `INTEGER = 1`)
    
    Finally, the `build_data_type_tables` class method must be called after all nodes are registered, such as at the end of the file defining them.
    This function performs the data type processing necessary to construct GP trees.
    
    The `GPTree` class is all that is necessary to interact with `GPNode`s, but the following functions are involved in this process:
    
    - `random_function(output_type, terminal=False, non_semi_terminal=False, non_terminal=False, branch=False,
                        num_children=None, forbidden_nodes=None)`
    
- `GPTree` is a subclass of [`BaseGenotype`](../evolution) that organizes genetic programming trees of `GPNode`s.
    Note that the tree itself is purely made of `GPNode`s, this class merely contains functions for interacting with the tree.
    The `execute` method takes in a `context` dictionary to be referenced by certain nodes, executes the whole tree, and returns the tree's return value.

    `GPTree`s take the following parameters for their dictionary initialization in `__init__`:
    - `"nodeType"` - A subclass of `GPNode` to be used in the tree, or the string name of that subclass, if it has been imported.
    - `"returnType"` - The return type of the tree, referencing the grammar of the `"nodeType"`.
    - `"minHeight"` - The minimum height of tree to randomly initialize.
    - `"maxHeight"` - The maximum height of tree to randomly initialize.
    - `"forbiddenNodes"` - Allows some otherwise valid GP nodes to be excluded from this tree; a list of node IDs, referencing the grammar of the `"nodeType"`.
    - `"idList"` - If provided, constructs the tree by expanding the given list of node IDs, referencing the grammar of the `"nodeType"`, otherwise the tree will be randomly generated.
    
    

- `GPNodeTypeRegistry` stores a global dictionary `name_lookup` mapping the string names of any imported GP node classes to the class object, allowing instantiation from a string description.
