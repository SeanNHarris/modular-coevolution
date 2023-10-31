from typing import Any, Optional, Literal, TypedDict, Type, Union

import abc
import math

from evolution.specialtypes import EvaluationID

MetricTypes = Union[float, str, list, dict]


class MetricConfiguration(TypedDict):
    """A `TypedDict` used to configure a metric or objective and describe how it should be handled."""
    name: str
    """A string key for storing this metric."""
    is_objective: bool
    """If true, this metric is considered an objective.
    Objectives must be numeric values.
    Objectives are reported to generators, such as fitness for standard evolutionary algorithms.
    Other metrics are just stored for logging and analysis purposes."""
    repeat_mode: Literal['replace', 'average', 'min', 'max']
    """How to handle multiple submissions of the same metric. The following modes are supported:
    - ``'replace'``: Overwrite the previous value with the new one.
    - ``'average'``: Record the mean of all submitted values. Must be a numeric type.
    - ``'min'``: Record the minimum of all submitted values. Must be a numeric type.
    - ``'max'``: Record the maximum of all submitted values. Must be a numeric type."""
    log_history: bool
    """If true, store a history of all submitted values for this metric. Avoid using this unnecessarily, as it can impact the size of the log file."""
    automatic: bool
    """If true, this metric will be automatically computed by the :class:`.BaseGenerator` and does not need to be submitted manually."""


class MetricSubmission(TypedDict, MetricConfiguration):
    """A `TypedDict` used to send metrics and objectives, and describe how they should be handled."""
    value: MetricTypes
    """The value to submit for this metric."""


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
        self.metrics = {}
        self.metric_statistics = {}
        self.metric_histories = {}
        self._objective_names = []
        self.evaluation_ids = []

    def submit_metric(self, submission: MetricSubmission) -> None:
        """Submit a metric value. This should typically be handled by :class:`.BaseObjectiveGenerator`.

        Args:
            submission: A :class:`.MetricSubmission` describing the metric to submit.

        """
        metric = submission['name']
        value = submission['value']

        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float) and submission['repeat_mode'] != 'replace':
            raise ValueError(f"Metric {metric} was submitted with repeat_mode={submission['repeat_mode']}, but is not a numeric value.")

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

            if submission['log_history']:
                self.metric_histories[metric].append(value)

    def set_objectives(self, objectives: dict[str, float]) -> None:
        raise NotImplementedError("This method has been removed. Use submit_metric instead.")

    def get_active_objectives(self):
        raise NotImplementedError("This method has been removed.")

    def get_fitness_modifier(self, raw_fitness):
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
            log_history=False
        ))

    def log_evaluation_id(self, evaluation_id: EvaluationID) -> None:
        """Log an evaluation id this individual participated in.

        Args:
            evaluation_id: The evaluation id to log.

        """
        self.evaluation_ids.append(evaluation_id)