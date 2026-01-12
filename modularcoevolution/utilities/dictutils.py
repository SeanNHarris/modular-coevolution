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

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

from typing import Sequence, Any


def deep_copy_dictionary(dictionary: dict) -> dict:
    """
    Create a deep copy of a dictionary. Unlike `copy.deepcopy`,
    this function will not copy any objects which are not dictionaries.

    Args:
        dictionary: The original dictionary.

    Returns:
        dict: A deep copy of the original dictionary.

    """
    new_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dictionary[key] = deep_copy_dictionary(value)
        else:
            new_dictionary[key] = value
    return new_dictionary


def deep_update_dictionary(dictionary: dict, update: dict, weak: bool = False) -> None:
    """
    Update a dictionary with another dictionary, recursively.

    Args:
        dictionary: The original dictionary.
        update: A dictionary of keys to replace corresponding keys in `dictionary`.
        weak: If True, existing values will not be overwritten.
    """
    for key, value in update.items():
        if isinstance(value, dict):
            if key not in dictionary:
                dictionary[key] = {}
            deep_update_dictionary(dictionary[key], value, weak=weak)
        else:
            if not weak or key not in dictionary:
                dictionary[key] = value


def set_config_value(config: dict, keys: Sequence[str], value: Any, weak: bool = False) -> None:
    """
    Set a value in a nested dictionary using a list of keys.
    Sub-dictionaries will be created if they do not already exist.
    Args:
        config: The dictionary to modify.
        keys: A sequence of keys to traverse the nested dictionary.
        value: The value to set at the final key.
        weak: If True, the value will not be set if the key already exists.
    """
    current_dict = config
    for key in keys[:-1]:
        if key not in current_dict:
            current_dict[key] = {}
        current_dict = current_dict[key]

    if not weak or keys[-1] not in current_dict:
        current_dict[keys[-1]] = value


def has_config_value(config: dict, keys: Sequence[str]) -> bool:
    """
    Check if a value in a nested dictionary exists.
    Args:
        config: The dictionary to read.
        keys: A sequence of keys to traverse the nested dictionary.
    Returns:
        True if a value exists at the given key sequence, False otherwise.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        else:
            current = current[key]
    return True


def get_config_value(config: dict, keys: Sequence[str]) -> Any:
    """
    Get a value in a nested dictionary using a list of keys.
    Args:
        config: The dictionary to read.
        keys: A sequence of keys to traverse the nested dictionary.
    Returns:
        The value of the nested dictionary at the given key sequence.
    Raises:
        ValueError: If one of the specified keys does not exist.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Key {key} could not be found.")
        else:
            current = current[key]
    return current


def flatten_dictionary(config: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """
    Flatten a nested dictionary.
    The resulting dictionary keys correspond to the paths of values in the nested dictionary structure.
    Args:
        config: The dictionary to flatten.
    Returns:
        A flat dictionary, where keys are a tuple of strings such that `result[(a, b, c)] == config[a][b][c]`.
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            sub_result = flatten_dictionary(value)
            for sub_key, sub_value in sub_result.items():
                new_key = (key, *sub_key)
                result[new_key] = sub_value
        else:
            new_key = (key,)
            result[new_key] = value
    return result


def strip_dictionary_layers(dictionary: Any, layers: list[int]) -> Any:
    """
    Remove redundant layers of a nested dictionary based on the specified layer numbers.

    Args:
        dictionary: The dictionary to strip.
            Non-dictionary values will be returned unaltered (for recursion purposes).
        layers: A list of layer numbers to remove.

    Returns:
        The input dictionary with the specified layers omitted.
        This may not be a dictionary, if all layers were removed.

    Raises:
        ValueError: If one of the specified layers contains more than one key.
    """
    if len(layers) == 0:
        return dictionary  # No more alterations necessary.
    if not isinstance(dictionary, dict):
        return dictionary  # Base case, reached the bottom of the nested dictionary.
    if layers[0] == 0:
        if len(dictionary) > 1:
            raise ValueError(f"Cannot strip layer from a dictionary with {len(dictionary)} keys.")
        subdictionary = next(iter(dictionary.values()))
        new_layers = [layer - 1 for layer in layers[1:]]
        return strip_dictionary_layers(subdictionary, new_layers)
    else:
        new_layers = [layer - 1 for layer in layers]
        new_dictionary = {}
        for key, value in dictionary.items():
            new_dictionary[key] = strip_dictionary_layers(value, new_layers)
        return new_dictionary


def purge_dictionary_by_type(dictionary: dict, purge_type: type) -> dict:
    """
    Return a copy of a nested dictionary which omits values of a given type.
    Args:
        dictionary: The dictionary to process.
        purge_type: The type of value to omit from the returned dictionary.
    Returns:
        A copy of the dictionary with values of the specified type omitted.
    """
    result = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result[key] = purge_dictionary_by_type(value, purge_type)
        elif not isinstance(value, purge_type):
            result[key] = value
    return result
