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

def deep_update_dictionary(dictionary: dict, update: dict) -> None:
    """
    Update a dictionary with another dictionary, recursively.

    Args:
        dictionary: The original dictionary.
        update: A dictionary of keys to replace corresponding keys in `dictionary`.

    """
    for key, value in update.items():
        if isinstance(value, dict):
            if key not in dictionary:
                dictionary[key] = {}
            deep_update_dictionary(dictionary[key], value)
        else:
            dictionary[key] = value


def set_config_value(config: dict, keys: Sequence[str], value: Any) -> None:
    """
    Set a value in a nested dictionary using a list of keys. If a sub-dictionary does not exist, it will be created.
    Args:
        config: The dictionary to modify.
        keys: A sequence of keys to traverse the nested dictionary.
        value: The value to set at the final key.
    """
    current_dict = config
    for key in keys[:-1]:
        if key not in current_dict:
            current_dict[key] = {}
        current_dict = current_dict[key]
    current_dict[keys[-1]] = value