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
