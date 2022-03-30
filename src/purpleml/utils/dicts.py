def recursive_items(dictionary, prefix=None):

    if prefix is None:
        prefix = []

    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield from recursive_items(value, prefix + [key])
        else:
            yield (prefix + [key], value)


def recursive_get(dictionary, keys, default=None, exception=False):    
    for k in keys:
        if k not in dictionary:
            if exception:
                raise KeyError("{} in {}".format(k, keys))
            else:
                return default 
        dictionary = dictionary[k]
    return dictionary


def recursive_set(dictionary, keys, value=None, default=None):
    if default is None:
        default = {}
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, default)
    dictionary[keys[-1]] = value


def recursive_merge(dict1, dict2, merge_lists=False, excludes=None):

    if excludes is None:
        excludes = []
    
    for k2, v2 in dict2.items():
        if k2 not in excludes:
            if isinstance(v2, dict) and k2 in dict1 and isinstance(dict1[k2], dict):
                recursive_merge(dict1[k2], v2, merge_lists=merge_lists, excludes=excludes)
            elif merge_lists \
                    and isinstance(v2, list) and k2 in dict1 \
                    and isinstance(dict1[k2], list) and len(v2) == len(dict1[k2]) \
                    and all([isinstance(e, dict) for e in dict1[k2]]) and all([ isinstance(e, dict) for e in v2]):
                dict1[k2] = [recursive_merge(e1, e2, merge_lists=merge_lists, excludes=excludes) for e1, e2 in zip(dict1[k2], v2)]
            else:
                dict1[k2] = v2
    
    return dict1


def flatten(dictionary, sep_level=".", sep_tuple="___", key_element_handler=None, value_handler=None, return_style=None):

    if return_style is None:
        return_style = "generator"

    if key_element_handler is None:
        key_element_handler = lambda e: sep_tuple.join(e) if isinstance(e, tuple) or isinstance(e, list) else e

    if value_handler is None:
        value_handler = lambda x: x

    def gen():
        for key, value in recursive_items(dictionary):
            yield sep_level.join([ key_element_handler(e) for e in key ]), value_handler(value)

    if return_style == "generator":
        return gen()
    elif return_style == "dict":
        return { key: value for key, value in gen() }
    elif return_style == "list":
        return [ (key, value) for key, value in gen() ]
    else:
        raise Exception("Unknown return style: {}".format(return_style))
    

def unflatten(input, sep_level=".", sep_tuple="___", key_parser=None, value_parser=None):

    if key_parser is None:
        def level_parser(level):
            split = level.split(sep_tuple)
            return split[0] if len(split) is 1 else tuple(split)

        def default_parser(key):
            levels = key.split(sep_level)
            if len(levels) is 1:
                return level_parser(levels[0])
            else:
                return [ level_parser(level) for level in levels  ]

        key_parser = default_parser

    if value_parser is None:
        value_parser = lambda v: v

    iterable = input
    if isinstance(input, dict):
        iterable = input.items()
    
    return_dict = {}
    for k, v in iterable:
        # print(key_parser(k), value_parser(v))
        recursive_set(return_dict, key_parser(k), value_parser(v))
    
    return return_dict