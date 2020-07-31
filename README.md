# ceads-evolution
CSSI CEADS-LIN Evolutionary Computation

The `ceads-evolution` repository contains general-purpose functions for evolutionary genotypes and agents used by both attackers and defenders, as well as the functions which coordinate evolution.
The main script for running evolution, `CoevolutionDriver.py` is located here.
An explanation of this repository's subdirectories follows:

### Evolution
[`Evolution`](/Evolution) contains the immediate components needed to run evolution or coevolution.

### GeneticProgramming
[`GeneticProgramming`](/GeneticProgramming) contains the `GPTree` class used by all genetic programming trees, and the base class for `GPNode`s.

### AlternateGenerators
[`AlternateGenerators`](/AlternateGenerators) contains some alternative genotype generators to the standard `EvolutionGenerator`.

### alternate_genotypes
[`alternate_genotypes`](/alternate_genotypes) contains genotypes other than the standard `GPTree`, including `MultipleGenotype` which can contain a set of other genotypes.

### diversity
[`diversity`](/diversity) contains functions related to population diversity and novelty calculation.

### postprocessing
[`postprocessing`](/postprocessing) contains data visualization functions to be run on log files.

### utilities
[`utilities`](/utilities) contains debugging tools, such as `LogReader` and tools for starting test evaluations.