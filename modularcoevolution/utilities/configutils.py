#  Copyright 2026 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import itertools
import logging
import tomllib
from typing import Any, Sequence

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import fileutils, dictutils


def parse_config(
        config_filename: str,
        merge_parameters: dict = None,
        experiment_type: type[BaseExperiment] = None
) -> dict[str, Any]:
    """Parse a configuration file and return a dictionary of parameters for each run.

    Args:
        config_filename: The filename of the configuration file to parse.
        merge_parameters: A dictionary of parameters to merge into the configuration file's parameters.
            Conflicting parameters will be overwritten.
        experiment_type: If provided, this experiment type will be saved to the parameter list for logging.
            This doesn't need to be provided if an experiment type is specified in the configuration file.

    Returns:
        A list containing a dictionary of parameters for each run.
    """
    config_path = fileutils.resolve_config_path(config_filename)
    with open(config_path, 'rb') as config_file:
        parameters = tomllib.load(config_file)

    if experiment_type is not None:
        experiment_type_string = f"{experiment_type.__module__}.{experiment_type.__name__}"
        dictutils.set_config_value(parameters, ('experiment_type',), experiment_type_string, weak=True)

    if merge_parameters is not None:
        dictutils.deep_update_dictionary(parameters, merge_parameters)

    return parameters


def generate_run_parameters(
        config: dict[str, Any],
        run_count: int,
        run_start: int = 0,
        merge_parameters: dict | list[dict] = None,
) -> list[dict[str, Any]]:
    """Parse a configuration file and return a dictionary of parameters for each run.

    Args:
        config: A config dictionary, such as the one returned by :func:`parse_config`.
        run_count: The total number of runs to perform, including runs skipped by the ``run_start`` argument.
        run_start: The run number to start at. Runs will end at the number specified by the ``run_amount`` argument.
        merge_parameters: A dictionary of parameters to merge into the configuration file's parameters,
            or a list of dictionaries per run to merge.
        experiment_type: If provided, this experiment type will be saved to the parameter list for logging.
            This doesn't need to be provided if an experiment type is specified in the configuration file.

    Returns:
        A list containing a dictionary of parameters for each run.
    """
    treatment_merge_parameters = _process_metaparameters(config)

    if merge_parameters is None:
        merge_parameters = {}
    if isinstance(merge_parameters, dict):
        merge_parameters = [merge_parameters] * run_count

    parameter_list = []

    if treatment_merge_parameters is None:
        for i in range(run_start, run_count):
            run_parameters = dictutils.deep_copy_dictionary(config)
            run_parameters['log_subfolder'] = f"{config['log_folder']}/Run {i}"
            dictutils.deep_update_dictionary(run_parameters, merge_parameters[i])

            parameter_list.append(run_parameters)
    else:
        for treatment in treatment_merge_parameters:
            meta_run_parameters = dictutils.deep_copy_dictionary(config)
            dictutils.deep_update_dictionary(meta_run_parameters, treatment)
            treatment_string = treatment['treatment_string']
            for i in range(run_count):
                run_parameters = dictutils.deep_copy_dictionary(meta_run_parameters)
                run_parameters['log_subfolder'] = f"{config['log_folder']}/{treatment_string}/Run {i}"
                dictutils.deep_update_dictionary(run_parameters, merge_parameters[i])

                parameter_list.append(run_parameters)

        logger = logging.getLogger(__name__)
        logger.info(f"Note: metaparameters in the config file are requesting {len(treatment_merge_parameters)} treatments.")
        logger.info(f"This will result in {len(parameter_list)} total runs.")
    return parameter_list


def _process_metaparameters(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Handle metaparameters such as combinations of parameter values to vary across runs."""
    meta_values = None
    meta_sets = None
    if 'meta_values' in config:
        meta_values = _process_meta_values(config)
    if 'meta_sets' in config:
        meta_sets = _process_meta_sets(config)

    if meta_values is not None and meta_sets is not None:
        result = []
        for meta_value_group, meta_set in itertools.product(meta_values, meta_sets):
            value_treatment_string = meta_value_group['treatment_string']
            set_treatment_string = meta_set['treatment_string']
            segments = set_treatment_string.split('-') + value_treatment_string.split('-')
            segments = [segment for segment in segments if segment != 'treatment']
            treatment_string = "treatment-" + "-".join(segments)

            merge_parameters = dictutils.deep_copy_dictionary(meta_set)
            dictutils.deep_update_dictionary(merge_parameters, meta_value_group)
            dictutils.set_config_value(merge_parameters, ('treatment_string',), treatment_string)
            result.append(merge_parameters)
        return result
    elif meta_values is not None:
        return meta_values
    elif meta_sets is not None:
        return meta_sets
    else:
        return None


def _process_meta_values(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Each metaparameter is a list of values for that parameter.
    The set of treatments generated is the Cartesian product of these lists of values."""
    if 'meta_values' not in config:
        return None

    meta_values = config['meta_values']
    flat_values = dictutils.flatten_dictionary(meta_values)

    for key, value in flat_values.items():
        if not isinstance(value, Sequence):
            path = ".".join(key)
            raise TypeError(f"Could not parse config metaparameters. {path} is not a list of values.")

    combinations = itertools.product(*flat_values.values())
    result = []
    for combination in combinations:
        value_strings = ['treatment']
        for index, value in enumerate(combination):
            if isinstance(value, str):
                value_strings.append(value)
            elif isinstance(value, int):
                value_strings.append(str(value))
            elif isinstance(value, float):
                value_strings.append(str(value).replace('.', '_'))
            else:
                value_index = list(flat_values.values()).index(value)
                value_strings.append(str(value_index))
        treatment_string = "-".join(value_strings)

        merge_config = {}
        dictutils.set_config_value(merge_config, ('treatment_string',), treatment_string)
        for key, value in zip(flat_values.keys(), combination):
            dictutils.set_config_value(merge_config, key, value)
        result.append(merge_config)

    return result


def _process_meta_sets(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Provide a list or dict of merge parameter dictionaries, each representing a treatment.
    A `treatment_string` parameter can be included to specify the name of the treatment for logging.
    Alternatively, if a dict was provided, the keys will be used as treatment strings."""
    if 'meta_sets' not in config:
        return None

    meta_sets = config['meta_sets']
    if not isinstance(meta_sets, list) and not isinstance(meta_sets, dict):
        raise TypeError("Could not parse config metaparameters. meta_sets must be a list of parameter dictionaries.")

    result = []

    if isinstance(meta_sets, dict):
        for treatment_string, meta_set in meta_sets.items():
            if not isinstance(meta_set, dict):
                raise TypeError(f"Could not parse config metaparameters. meta_sets['{treatment_string}'] is not a parameter dictionary.")
            merge_config = meta_set.copy()
            dictutils.set_config_value(merge_config, ('treatment_string',), treatment_string)
            result.append(merge_config)
        return result

    for index, meta_set in enumerate(meta_sets):
        if not isinstance(meta_set, dict):
            raise TypeError(f"Could not parse config metaparameters. meta_sets[{index}] is not a parameter dictionary.")

        merge_config = meta_set.copy()

        if 'treatment_string' not in merge_config:
            value_strings = ['treatment']
            flat_values = dictutils.flatten_dictionary(merge_config)
            for key, value in flat_values.items():
                if isinstance(value, str):
                    value_strings.append(value)
                elif isinstance(value, int):
                    value_strings.append(str(value))
                elif isinstance(value, float):
                    value_strings.append(str(value).replace('.', '_'))
                else:
                    # No clear value to put in the filename.
                    # A treatment string should be provided for weird parameters like this.
                    value_strings.append('x')
            treatment_string = "-".join(value_strings)
            dictutils.set_config_value(merge_config, ('treatment_string',), treatment_string)

        result.append(merge_config)
    return result


def experiment_from_config(config_parameters: dict[str, Any]) -> BaseExperiment:
    """Create an experiment from a loaded configuration file.

    Args:
        config_parameters: Configuration parameters as returned by :func:`parse_config`.

    Returns:
        An experiment object with the type and parameters given in the config dictionary.
    """
    # Should get_experiment_type be moved?
    experiment_type = postprocessingutils.get_experiment_type(config_parameters)
    return experiment_type(config_parameters)


def run_parameters_from_logs(experiment_path_str: str) -> list[dict[str, Any]]:
    """Get a list of configuration dictionaries for each run within a given experiment folder.
    This should be comparable to the configuration dictionaries generated by :func:`generate_run_parameters` for the same experiment.

    Args:
        experiment_path_str: The path to the experiment folder within the logs directory.

    Returns:
        A list of configuration dictionaries for each run within the given experiment folder.
    """

    run_parameters = []

    treatment_paths = fileutils.get_treatment_paths(experiment_path_str)  # Works even if there are no sub-experiments.
    for treatment_path, _ in treatment_paths.items():
        run_paths = fileutils.get_run_paths(treatment_path)
        for run_number, run_path in run_paths.items():
            parameters = postprocessingutils.load_run_config(run_path)
            run_parameters.append(parameters)

    return run_parameters


def merge_run_parameters(primary_run_parameters: list[dict[str, Any]], secondary_run_parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge two lists of run parameters, such as those generated by :func:`generate_run_parameters` and :func:`run_parameters_from_logs`.
    Handles treatments by treatment string and runs by run index.

    Args:
        primary_run_parameters: The primary list of run parameters. Takes priority over the secondary parameters.
        secondary_run_parameters: The secondary list of run parameters. Only used if there are missing parameters in the primary parameters.

    Returns:
        A list of run parameters comparable to those generated by :func:`generate_run_parameters`, merging the parameters from the primary and secondary lists.
    """

    primary_parameters_by_treatment = {}
    secondary_parameters_by_treatment = {}
    for run_parameters, treatment_map in [(primary_run_parameters, primary_parameters_by_treatment), (secondary_run_parameters, secondary_parameters_by_treatment)]:
        for parameters in run_parameters:
            run_identifiers = []
            if 'treatment_string' in parameters:
                run_identifiers.append(parameters['treatment_string'])

            run_folder = parameters['log_subfolder'].split('/')[-1]
            run_identifiers.append(run_folder)

            run_identifier = "/".join(run_identifiers)
            treatment_map[run_identifier] = parameters

    merged_run_parameters = []
    for run_identifier, primary_parameters in primary_parameters_by_treatment.items():
        if run_identifier not in secondary_parameters_by_treatment:
            raise ValueError(f"Primary run parameters contain a run \"{run_identifier}\" that is not present in the secondary run parameters.")
        secondary_parameters = secondary_parameters_by_treatment[run_identifier]
        merged_parameters = primary_parameters.copy()
        dictutils.deep_update_dictionary(merged_parameters, secondary_parameters, weak=True)
        merged_run_parameters.append(merged_parameters)

    return merged_run_parameters

