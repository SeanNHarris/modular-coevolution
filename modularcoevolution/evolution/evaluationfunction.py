from typing import Union, Any, Sequence, Callable

ResultsType = Union[dict[str, Any], list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]
EvaluateType = Callable[[Sequence, Any, ...], ResultsType]
"""
    Args:
        agents: An ordered list of agents, e.g. [attacker, defender]
        **kwargs: Any number of keyword arguments that will be passed by ``world_kwargs``.
    Returns:
        One of three result formats:
        #. A dictionary of results, e.g. {'attacker_score': 0.5, 'defender_cost': 0.3}
        #. A list of dictionaries of results, one for each player in the order of ``agents``.
        #. Two values: a list of dictionaries of agent-specific results, and a dictionary of shared additional results.
        The alternative formats are useful when drawing multiple agents from the same population.
"""


class EvaluationFunction:
    """A wrapper for an evaluation function which stores information about the structure of the 'game' modeled by the function and """
    evaluate: EvaluateType
    """The evaluation function to use."""

    num_players: int
    """The number of players in the game."""