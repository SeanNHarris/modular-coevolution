from typing import Literal, TypedDict, Union

import abc
import math

import numpy

from modularcoevolution.utilities.specialtypes import EvaluationID, GenotypeID

MetricTypes = Union[float, str, list, dict, numpy.ndarray]


class MetricConfiguration(TypedDict):
    """A `TypedDict` used to configure a metric or objective and describe how it should be handled."""
    name: str
    """A string key for storing this metric."""
    is_objective: bool
    """If true, this metric is considered an objective.
    Objectives must be numeric values.
    Objectives are reported to generators, such as fitness for standard evolutionary algorithms.
    Other metrics are just stored for logging and analysis purposes."""
    repeat_mode: Literal['replace', 'average', 'min', 'max', 'sum']
    """How to handle multiple submissions of the same metric. The following modes are supported:
    - ``'replace'``: Overwrite the previous value with the new one.
    - ``'average'``: Record the mean of all submitted values. Must be a numeric type.
    - ``'min'``: Record the minimum of all submitted values. Must be a numeric type.
    - ``'max'``: Record the maximum of all submitted values. Must be a numeric type.
    - ``'sum'``: Record the sum of all submitted values. Must be a numeric type."""
    log_history: bool
    """If true, store a history of all submitted values for this metric. Avoid using this unnecessarily, as it can impact the size of the log file."""
    automatic: bool
    """If true, this metric will be automatically computed by the :class:`.BaseGenerator` and does not need to be submitted manually."""
    add_fitness_modifier: bool
    """If true, the individual's :meth:`.get_fitness_modifier` result will be added to this metric (e.g. for parsimony pressure)."""


class MetricSubmission(MetricConfiguration, TypedDict, total=False):
    """A `TypedDict` used to send metrics and objectives, and describe how they should be handled."""
    value: MetricTypes
    """The value to submit for this metric."""
    opponents: list[GenotypeID]
    """The genotype IDs of the opponents this metric was evaluated against, if applicable."""


class MetricStatistics(TypedDict, total=False):
    """A `TypedDict` storing running statistics about submitted metric values."""
    count: int
    """The number of times this metric has been submitted."""
    mean: float
    """The mean value of submitted metric values."""
    std_dev_intermediate: float
    """An intermediate value used to calculate the standard deviation as a running total via Welford's Algorithm."""
    standard_deviation: float
    """The standard deviation of this metric."""
    minimum: float
    """The minimum observed value of this metric."""
    maximum: float
    """The maximum observed value of this metric."""


# TODO: Consider refactoring the objective tracker into a property of a genotype, rather than a superclass.
# Pros:
# - This is more natural, and helps in cases where we need to manipulate the objective tracker separate from the genotype.
# - Some methods currently need to be named to specify that they relate to the tracker specifically.
# Cons:
# - This change will affect most of the codebase, and require a lot of effort refactoring.
# - Accessing objective tracker properties will be slightly more verbose.
# - We still will want a superclass/interface for BaseGenotype to handle non-genotype objects with a data tracker.
#   - (This ability is currently unused, but we want it to be available for future use cases such as optimal agents.)
class BaseObjectiveTracker(metaclass=abc.ABCMeta):
    """
    A base class for anything that needs to track objectives or fitness. Formerly part of :class:`.BaseGenotype`.

    This class can store a variety of data metrics about the individual, some of which are flagged as objectives.
    Objectives are used by :class:`.BaseObjectiveGenerator` to calculate fitness values.
    Other metrics are only logged by the :class:`.DataCollector` or other logging tools.
    """

    metrics: dict[str, MetricTypes]
    """A dictionary of metric values tracked by this individual, by name."""
    metric_statistics: dict[str, MetricStatistics]
    """The statistics tracked for each metric, by name."""
    metric_histories: dict[str, list[MetricTypes]]
    """A history of all values submitted for each metric, by name. Only stored if ``log_history=True`` was passed to :meth:`.submit_metric`."""
    _objective_names: list[str]
    """The names of metrics which were submitted as objectives."""
    evaluation_ids: list[EvaluationID]
    """A list of evaluation IDs this individual participated in."""
    opponent_trackers: dict[GenotypeID, 'BaseObjectiveTracker']
    """A dictionary tracking metrics specific to each opponent this individual has faced."""

    @property
    def objectives(self) -> dict[str, float]:
        """A dictionary of objective values, by metric name."""
        return {name: self.metrics[name] for name in self._objective_names}

    @property
    def is_evaluated(self) -> bool:
        """Has this individual been evaluated at least once? If ``False``, it will not have valid objectives."""
        return len(self._objective_names) > 0

    @property
    def fitness(self) -> float:
        """An objective named ``fitness``, given as a shortcut for single-objective evolution.
        Use :meth:`.set_fitness` as a shortcut to set this value."""
        if 'fitness' not in self.objectives:
            if len(self.objectives) == 1:
                bad_objective = next(iter(self.objectives))
                raise KeyError(f"This individual does not have an objective named 'fitness', but a single-objective class is being used which requires it. You may need to rename the single objective {bad_objective} to 'fitness'.")
            raise KeyError("This individual does not have an objective named 'fitness', but a single-objective class is being used which requires it.")

        return self.objectives['fitness']

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reset_objective_tracker()

    def reset_objective_tracker(self):
        """Reset and clear all metrics and statistics."""
        self.metrics = {}
        self.metric_statistics = {}
        self.metric_histories = {}
        self._objective_names = []
        self.evaluation_ids = []
        self.opponent_trackers = {}

    def submit_metric(self, submission: MetricSubmission, prevent_recursion: bool = False) -> None:
        """Submit a metric value. This should typically be handled by :class:`.BaseObjectiveGenerator`.

        Args:
            submission: A :class:`.MetricSubmission` describing the metric to submit.
            prevent_recursion: Set to true if this method is being called from a parent :class:`.BaseObjectiveTracker`
                to prevent infinite recursion.

        """
        metric = submission['name']
        value = submission['value']

        if isinstance(value, int):
            value = float(value)
        if not (isinstance(value, float) or isinstance(value, numpy.ndarray)) and submission['repeat_mode'] != 'replace':
            raise ValueError(f"Metric {metric} was submitted with repeat_mode={submission['repeat_mode']}, but is not a numeric value.")

        # Add fitness modifier
        if submission['add_fitness_modifier']:
            if not isinstance(value, float):
                raise ValueError(
                    f"Metric {metric} was submitted with add_fitness_modifier=True, but is not a numeric value.")
            value += self.get_fitness_modifier(value)

        new_metric = metric not in self.metrics
        if new_metric:
            self.metrics[metric] = value
            if isinstance(value, float):
                self.metric_statistics[metric] = {
                    "count": 0,
                    "mean": 0,
                    "std_dev_intermediate": 0,
                    "standard_deviation": 0,
                    "minimum": math.inf,
                    "maximum": -math.inf
                }
            else:
                self.metric_statistics[metric] = {
                    "count": 0
                }
            if submission['is_objective']:
                if not isinstance(value, float):
                    raise ValueError(f"Objective {metric} must be a numeric value.")
                self._objective_names.append(metric)
            if submission['log_history']:
                self.metric_histories[metric] = [value]
        else:
            if submission['is_objective'] != (metric in self._objective_names):
                raise ValueError(f"Metric {metric} was submitted with is_objective={submission['is_objective']}, but was previously submitted with is_objective={metric in self._objective_names}.")
            if submission['log_history'] != (metric in self.metric_histories):
                raise ValueError(f"Metric {metric} was submitted with log_history={submission['log_history']}, but was previously submitted with log_history={metric in self.metric_histories}.")

        # Update statistics
        self.metric_statistics[metric]["count"] += 1
        # Update numeric statistics when relevant
        if isinstance(value, float):
            previous_mean = self.metric_statistics[metric]["mean"]
            total = self.metric_statistics[metric]["mean"] * (self.metric_statistics[metric]["count"] - 1) + value
            self.metric_statistics[metric]["mean"] = total / self.metric_statistics[metric]["count"]
            self.metric_statistics[metric]["std_dev_intermediate"] += (value - previous_mean) * (value - self.metric_statistics[metric]["mean"])
            self.metric_statistics[metric]["standard_deviation"] = math.sqrt(self.metric_statistics[metric]["std_dev_intermediate"] / self.metric_statistics[metric]["count"])
            self.metric_statistics[metric]["minimum"] = min(self.metric_statistics[metric]["minimum"], value)
            self.metric_statistics[metric]["maximum"] = max(self.metric_statistics[metric]["maximum"], value)

        # Update metric value
        if not new_metric:
            if submission['repeat_mode'] == 'replace':
                self.metrics[metric] = value
            elif submission['repeat_mode'] == 'average':
                self.metrics[metric] = self.metric_statistics[metric]["mean"]
            elif submission['repeat_mode'] == 'min':
                self.metrics[metric] = self.metric_statistics[metric]["minimum"]
            elif submission['repeat_mode'] == 'max':
                self.metrics[metric] = self.metric_statistics[metric]["maximum"]
            elif submission['repeat_mode'] == 'sum':
                self.metrics[metric] += value

            if submission['log_history']:
                self.metric_histories[metric].append(value)

        # Update opponent-specific metrics
        if not prevent_recursion and 'opponents' in submission:
            for opponent in submission['opponents']:
                if opponent not in self.opponent_trackers:
                    self.opponent_trackers[opponent] = BaseObjectiveTracker()
                self.opponent_trackers[opponent].submit_metric(submission, prevent_recursion=True)

    def set_objectives(self, objectives: dict[str, float]) -> None:
        raise NotImplementedError("This method has been removed. Use submit_metric instead.")

    def get_active_objectives(self):
        raise NotImplementedError("This method has been removed.")

    def get_fitness_modifier(self, raw_fitness: float) -> float:
        """Return a fitness modifier to be added to the given raw fitness value.
        This is most commonly used for parsimony pressure.

        Args:
            raw_fitness: The raw value of the metric.

        Returns:
            A value to be added to the raw fitness value. Use a negative value for a penalty.
        """
        return 0

    def set_fitness(self, fitness: float, average: bool = False) -> None:
        """Shortcut for single-objective evolution. Sets an objective named ``'fitness'``.

        Args:
            fitness: A fitness value to store.
            average: If True, the stored fitness value will be averaged across values sent over all calls of this
                method. If False, the stored fitness value will be set to the given value, overwriting previous values.

        """
        self.submit_metric(MetricSubmission(
            name='fitness',
            value=fitness,
            is_objective=True,
            repeat_mode='average' if average else 'replace',
            log_history=False,
            add_fitness_modifier=True
        ))

    def log_evaluation_id(self, evaluation_id: EvaluationID) -> None:
        """Log an evaluation id this individual participated in.

        Args:
            evaluation_id: The evaluation id to log.

        """
        self.evaluation_ids.append(evaluation_id)

    def get_opponents(self) -> list[GenotypeID]:
        """Return a list of genotype IDs of opponents this individual has been evaluated against."""
        return list(self.opponent_trackers.keys())

    def get_opponent_metrics(self, opponent: GenotypeID) -> dict[str, float]:
        """Return the metrics of this individual against a specific opponent.

        Args:
            opponent: The genotype ID of the opponent.

        Returns:
            A dictionary of metric values, by metric name.

        """
        if opponent not in self.opponent_trackers:
            raise KeyError(f"This individual has no evaluations against individual {opponent}.")
        return self.opponent_trackers[opponent].metrics


def compute_shared_objectives(individuals: list[BaseObjectiveTracker], opponent_id: GenotypeID, objective: str = 'fitness', total: float = 1) -> list[float]:
    """Compute the shared fitness, or the equivalent for another objective,
    for a group of individuals against a common opponent.

    Shared fitness assigns a "bounty" to an opponent, which is split among individuals that score against it.
    In this implementation, the bounty is split proportionally to the objective score,
    scaled so that the worst individual gets a score of zero.

    Args:
        individuals: A list of individuals to compute shared objective score for.
        opponent_id: The genotype ID of the opponent.
        objective: The name of the base objective to compute shared objective score for.
        total: The total amount of shared objective score to distribute.

    Returns:
        A list of shared objective scores, one for each individual.
    """
    objective_values = []
    min_value = None
    for individual in individuals:
        if objective not in individual.metrics or opponent_id not in individual.opponent_trackers:
            objective_values.append(math.nan)
        else:
            objective_value = individual.get_opponent_metrics(opponent_id)[objective]
            objective_values.append(objective_value)
            if min_value is None or objective_value < min_value:
                min_value = objective_value
    if min_value is None:
        min_value = 0
    for index, value in enumerate(objective_values):
        if math.isnan(value):
            objective_values[index] = min_value
    weights = [value - min_value for value in objective_values]
    weight_sum = sum(weights)
    if weight_sum == 0:
        weight_sum = 1  # Prevent division by zero; this doesn't change the result because all weights are zero
    scores = [total * weight / weight_sum for weight in weights]
    return scores


