from modularcoevolution.evolution.wrappers.coevolution import Coevolution
from modularcoevolution.evolution.wrappers.elocoevolution import EloCoevolution
from modularcoevolution.evolution.wrappers.evolutionwrapper import EvolutionWrapper
from modularcoevolution.evolution.wrappers.similarstrengthcoevolution import SimilarStrengthCoevolution
from modularcoevolution.evolution.wrappers.staticelocoevolution import StaticEloCoevolution
try:
    from modularcoevolution.evolution.wrappers.alpharankcoevolution import *
except ModuleNotFoundError:
    pass