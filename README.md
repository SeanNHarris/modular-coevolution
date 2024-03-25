# modular-coevolution
A competitive coevolution library by Sean Harris

The `modular-coevolution` repository contains general-purpose functions for evolutionary genotypes and agents, as well as the functions which coordinate evolution.
The main script for running evolution, `CoevolutionDriver.py` is located here.
An explanation of this repository's subdirectories follows:

### Evolution
[`Evolution`](/modularcoevolution/evolution) contains the immediate components needed to run evolution or coevolution. The `Evolution` readme explains the general structure of classes used for evolution.

### GeneticProgramming
[`GeneticProgramming`](/modularcoevolution/genotypes/geneticprogramming) contains the `GPTree` class used by all genetic programming trees, and the base class for `GPNode`s.

### AlternateGenerators
[`AlternateGenerators`](/modularcoevolution/alternategenerators) contains some alternative genotype generators to the standard `EvolutionGenerator`.

### AlternateGenotypes
[`AlternateGenotypes`](/modularcoevolution/alternategenotypes) contains genotypes other than the standard `GPTree`, including the basic `LinearGenotype`, and `MultipleGenotype` which can contain a set of other genotypes.

### Diversity
[`Diversity`](/modularcoevolution/genotypes/diversity) contains functions related to population diversity and novelty calculation.

### Postprocessing
[`Postprocessing`](/modularcoevolution/postprocessing) contains data postprocessing functions to be run on log files, such as generating CIAO/master tournament plots.

### Utilities
[`Utilities`](/modularcoevolution/utilities) contains miscellaneous helper functions for things like debugging.