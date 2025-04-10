#  Copyright 2025 BONSAI Lab at Auburn University
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

import tomllib
from typing import Any

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import fileutils, dictutils


def parse_config(
        config_filename: str,
        merge_parameters: dict = None,
        experiment_type: BaseExperiment = None
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
        config_filename: str,
        run_count: int,
        run_start: int = 0,
        merge_parameters: dict | list[dict] = None,
        experiment_type: type[BaseExperiment] = None
) -> list[dict[str, Any]]:
    """Parse a configuration file and return a dictionary of parameters for each run.

    Args:
        config_filename: The filename of the configuration file to parse.
        run_count: The total number of runs to perform, including runs skipped by the ``run_start`` argument.
        run_start: The run number to start at. Runs will end at the number specified by the ``run_amount`` argument.
        merge_parameters: A dictionary of parameters to merge into the configuration file's parameters,
            or a list of dictionaries per run to merge.
        experiment_type: If provided, this experiment type will be saved to the parameter list for logging.
            This doesn't need to be provided if an experiment type is specified in the configuration file.

    Returns:
        A list containing a dictionary of parameters for each run.
    """

    # Note: we don't pass in merge_parameters here, because we want to merge them separately per-run.
    base_parameters = parse_config(config_filename, experiment_type=experiment_type)

    if merge_parameters is None:
        merge_parameters = {}
    if isinstance(merge_parameters, dict):
        merge_parameters = [merge_parameters] * run_count

    parameter_list = []
    for i in range(run_start, run_count):
        run_parameters = dictutils.deep_copy_dictionary(base_parameters)
        run_parameters['log_subfolder'] = f"{base_parameters['log_folder']}/Run {i}"
        dictutils.deep_update_dictionary(run_parameters, merge_parameters[i])

        parameter_list.append(run_parameters)
    return parameter_list


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