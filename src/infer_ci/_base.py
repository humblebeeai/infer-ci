# Standard libraries
import pprint
import logging
from typing import Any

# Third-party libraries
from pydantic import validate_call

# Internal modules
from .__version__ import __version__
from . import _utils as utils
from .config import MyClassConfigPM


logger = logging.getLogger(__name__)


class MyClass:
    """A class to perform some operations on a list of float items.

    Attributes:
        items  (list[float]    ): List of float items to be processed.
        config (MyClassConfigPM): Configuration for the module.

    Methods:
        run(): Method to clean the items based on the threshold value.

    """

    @validate_call
    def __init__(
        self,
        items: list[float] | None = None,
        config: MyClassConfigPM | dict[str, Any] | None = None,
        auto_run: bool = False,
        **kwargs,
    ) -> None:
        """Initializer method for the MyClass class.

        Args:
            items  (list[float] | None                     , optional): List of float items to be processed.
                                                                            Defaults to None.
            config (MyClassConfigPM | dict[str, Any] | None, optional): Configuration for the module. Defaults to None.
        """

        logger.debug(
            f"Initializing <{self.__class__.__name__}> object with '{__version__}' version..."
        )
        if not config:
            config = MyClassConfigPM()

        self.config = config
        if kwargs:
            self.config = self.config.model_copy(update=kwargs)

        if items:
            self.items = items
        logger.debug(
            f"Initialized <{self.__class__.__name__}> object with '{__version__}' version."
        )

        if auto_run:
            self.run()

    @validate_call
    def run(
        self,
        items: list[float] | None = None,
        threshold: float | None = None,
    ) -> list[float]:
        """Method to clean the items based on the threshold value.

        Args:
            items     (list[float] | None, optional): List of float items to be processed. Defaults to None.
            threshold (float | None      , optional): Threshold value for the cleaning process. Defaults to None.

        Raises:
            RuntimeError: If `items` attribute is not set.

        Returns:
            list[float]: List of cleaned items.
        """

        if items:
            self.items = items

        if not hasattr(self, "items"):
            raise RuntimeError(
                "`items` attribute is not set, must provide a list of float items to be processed!"
            )

        if not threshold:
            threshold = self.config.threshold

        logger.debug(f"Cleaning items with threshold '{threshold}'...")
        _clean_items = []
        for _item in self.items:
            if threshold <= _item:
                _clean_items.append(_item)
            else:
                logger.debug(
                    f"Item '{_item}' is below the threshold '{threshold}', removing it..."
                )

        logger.debug("Successfully cleaned items.")

        self.items = _clean_items
        return self.items

    # ATTRIBUTES
    # config
    @property
    def config(self) -> MyClassConfigPM:
        try:
            return self.__config
        except AttributeError:
            self.__config = MyClassConfigPM()

        return self.__config

    @config.setter
    def config(self, config: MyClassConfigPM | dict[str, Any]) -> None:
        if (not isinstance(config, MyClassConfigPM)) and (not isinstance(config, dict)):
            raise TypeError(
                f"`config` attribute type {type(config)} is invalid, must be a <class 'MyClassConfigPM'> or <dict>!"
            )

        if isinstance(config, dict):
            config = MyClassConfigPM(**config)
        elif isinstance(config, MyClassConfigPM):
            config = config.model_copy(deep=True)

        self.__config = config

    # config

    # items
    @property
    def items(self) -> list[float]:
        try:
            return self.__items
        except AttributeError:
            raise AttributeError("`items` attribute is not set!")

    @items.setter
    def items(self, items: list[float]) -> None:
        if not isinstance(items, list):
            raise TypeError(
                f"`items` attribute type {type(items)} is invalid, must be a <class 'list'>!"
            )

        if (len(items) < self.config.min_length) or (
            self.config.max_length < len(items)
        ):
            raise ValueError(
                f"`items` attribute length '{len(items)}' is too short or too long, "
                f"must be between '{self.config.min_length}' and '{self.config.max_length}'!"
            )

        for _item in items:
            if not isinstance(_item, float):
                raise TypeError(
                    f"`items` attribute item type {type(_item)} is invalid, must be a <float>!"
                )

            if (_item < self.config.min_value) or (self.config.max_value < _item):
                raise ValueError(
                    f"`items` attribute item value '{_item}' is not in the allowed range, "
                    f"must be between '{self.config.min_value}' and '{self.config.max_value}'!"
                )

        self.__items = items

    # items
    # ATTRIBUTES

    # METHOD OVERRIDING
    def __str__(self):
        _self_dict = utils.clean_obj_dict(self.__dict__, self.__class__.__name__)
        _self_str = f"{self.__class__.__name__}: {pprint.pformat(_self_dict)}"
        return _self_str

    def __repr__(self):
        _self_repr = utils.obj_to_repr(self)
        return _self_repr

    # METHOD OVERRIDING


__all__ = ["MyClass"]
