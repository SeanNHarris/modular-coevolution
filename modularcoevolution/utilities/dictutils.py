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
            deep_update_dictionary(dictionary[key], value)
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