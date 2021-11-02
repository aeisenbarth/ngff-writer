from typing import Any, Callable, Dict, Sequence, Union


def nested_dict_apply_fn_at_path(
    nested_dict: Dict[str, Union[Any, dict]],
    key_path: Sequence[Union[str, int]],
    fn: Callable[[dict, str], None],
    create: bool = True,
) -> Any:
    """
    Applies a function at a specific node of a nested dictionary.

    Args:
        nested_dict: A nested Python dictionary
        key_path: A list of all keys up to the node (including the node's key).
        fn: A function that receives two arguments, the node's parent dictionary node and the node's
            key.
        create: Whether to create missing keys by inserting a new dictionary. Defaults to True.
    """
    if not (isinstance(key_path, Sequence) and all(isinstance(k, (str, int)) for k in key_path)):
        raise ValueError("Argument key_path must only contain strings of integers.")
    node = nested_dict
    for i, key in reversed(list(enumerate(reversed(key_path)))):
        if isinstance(node, dict) and key not in node:
            if create:
                node[key] = {}
            else:
                raise KeyError(f"{key_path}")
        elif isinstance(node, list) and len(node) <= key:
            raise KeyError(f"{key_path}")
        if i == 0:  # last key in path == the desired node
            return fn(node, key)
        else:
            node = node[key]  # continue
