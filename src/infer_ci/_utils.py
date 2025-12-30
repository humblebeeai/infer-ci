import logging

from pydantic import validate_call


logger = logging.getLogger(__name__)


@validate_call
def clean_obj_dict(obj_dict: dict, cls_name: str) -> dict:
    """Clean class name from object.__dict__ for str(object).

    Args:
        obj_dict (dict): Object dictionary by object.__dict__.
        cls_name (str ): Class name by cls.__name__.

    Returns:
        dict: Clean object dictionary.
    """

    try:
        if not obj_dict:
            raise ValueError("'obj_dict' argument value is empty!")

        if not cls_name:
            raise ValueError("'cls_name' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    _self_dict = obj_dict.copy()
    for _key in _self_dict.copy():
        _class_prefix = f"_{cls_name}__"
        if _key.startswith(_class_prefix):
            _new_key = _key.replace(_class_prefix, "")
            _self_dict[_new_key] = _self_dict.pop(_key)

    return _self_dict


@validate_call(config={"arbitrary_types_allowed": True})
def obj_to_repr(obj: object) -> str:
    """Modifying object default repr() to custom info.

    Args:
        obj (object): Any python object.

    Returns:
        str: String for repr() method.
    """

    try:
        if not obj:
            raise ValueError("'obj' argument value is empty!")

    except ValueError as err:
        logger.error(err)
        raise

    _self_repr = (
        f"<{obj.__class__.__module__}.{obj.__class__.__name__} object at {hex(id(obj))}: "
        + "{"
        + f"{str(dir(obj)).replace('[', '').replace(']', '')}"
        + "}>"
    )
    return _self_repr


__all__ = [
    "clean_obj_dict",
    "obj_to_repr",
]
