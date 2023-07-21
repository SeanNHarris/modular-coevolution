from modularcoevolution.evolution.baseobjectivetracker import BaseObjectiveTracker, MetricConfiguration, MetricTypes, \
    MetricSubmission
from modularcoevolution.evolution.specialtypes import GenotypeID, EvaluationID
from modularcoevolution.evolution.generators.basegenerator import BaseGenerator

from typing import Callable, Any, Union

import abc


MetricFunction = Callable[[dict[str, Any]], MetricTypes]


class BaseObjectiveGenerator(BaseGenerator, metaclass=abc.ABCMeta):
    """A :class:`.BaseGenerator` with standardized handling of objectives.

    When extending this class, the

    """

    metric_configurations: dict[str, MetricConfiguration]
    """A dictionary of registered metric configurations, keyed by metric name."""
    metric_functions: dict[str, Union[MetricFunction, str]]
    """A dictionary of registered metric functions, keyed by metric name.
    A string can be used instead of a function as a shortcut to return the value of a key in the input dictionary."""

    def __init__(self, *args, **kwargs):
        self.metric_configurations = {}
        self.metric_functions = {}

    @abc.abstractmethod
    def get_genotype_with_id(self, agent_id) -> BaseObjectiveTracker:
        """Return the agent parameters associated with the given ID.

        To work with the implementation of :meth:`.set_objectives` here, this must return a subclass of
        :class:`.BaseObjectiveTracker`. If no complex genotype object is required, note that a class can inherit from
        :class:`.BaseObjectiveTracker` and :class:`.BaseAgent` simultaneously.

        Args:
            agent_id: The ID of the agent parameter set being requested.

        Returns: The agent parameter set associated with the ID ``agent_id``.

        """
        pass

    def register_metric(self,
                        metric_configuration: MetricConfiguration,
                        metric_function: Union[MetricFunction, str]) -> None:
        """Register a metric with this generator.

        Args:
            metric_configuration: The metric to register.
            metric_function: A function which computes the metric from the dictionary of evaluation results.
                Alternatively, a string key in the dictionary of evaluation results which contains the metric value.

        """

        metric_name = metric_configuration['name']
        self.metric_configurations[metric_name] = metric_configuration
        self.metric_functions[metric_name] = metric_function

    def submit_metric(self, agent_id: GenotypeID, metric_name: str, metric_value: MetricTypes) -> None:
        """Submit a metric value for an agent to store.
        The metric must have been previously registered using :meth:`.register_metric`.

        Args:
            agent_id: The ID of the agent associated with the metric.
            metric_name: The name of the metric.
            metric_value: The value of the metric.

        """
        if metric_name not in self.metric_configurations:
            raise ValueError(f"Metric {metric_name} not registered.")

        metric_submission: MetricSubmission = self.metric_configurations[metric_name].copy()
        metric_submission['value'] = metric_value
        self.get_genotype_with_id(agent_id).submit_metric(metric_submission)

    def compute_metric(self, metric_name: str, evaluation_results: dict[str, Any]) -> MetricTypes:
        """Compute the requested metric from the given evaluation results.
        The metric must have been previously registered using :meth:`.register_metric`.

        Args:
            metric_name: The name of the metric.
            evaluation_results: The results of the evaluation.

        Returns:
            The value of the metric.

        """

        if metric_name not in self.metric_configurations:
            raise ValueError(f"Metric {metric_name} not registered.")

        if isinstance(self.metric_functions[metric_name], str):
            return evaluation_results[self.metric_functions[metric_name]]
        else:
            return self.metric_functions[metric_name](evaluation_results)



    def submit_evaluation(self, agent_id: GenotypeID, evaluation_id: EvaluationID, evaluation_results: dict[str, Any]) -> None:
        """Called by a :class:`.BaseEvolutionWrapper` to record objectives and metrics from evaluation results
        for the agent with given index.

        Args:
            agent_id: The index of the agent associated with the evaluation results.
            evaluation_id: The ID of the evaluation.
            evaluation_results: The results of the evaluation.

        """

        for metric_name in self.metric_configurations:
            self.submit_metric(agent_id, metric_name, self.compute_metric(metric_name, evaluation_results))

    def set_objectives(self, agent_id: GenotypeID, objectives: dict[str, float], opponent: GenotypeID = None, evaluation_id: EvaluationID = None) -> None:
        raise NotImplementedError("This method has been removed. Use submit_evaluation instead.")
