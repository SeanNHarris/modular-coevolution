import abc


class GPNodeTypeRegistry(abc.ABCMeta):  # Need to extend this to prevent conflicts, since GPNodes use ABCMeta
    name_lookup = dict()

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name not in mcs.name_lookup:
            mcs.name_lookup[name] = cls
        return cls