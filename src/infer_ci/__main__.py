# Standard libraries
import os
import sys
import logging

# Third-party libraries
from dotenv import load_dotenv

# Internal modules
from .config import MyClassCliConfig
from ._base import MyClass


load_dotenv(dotenv_path=".env", override=True)
logger = logging.getLogger(__name__)


def main() -> None:
    _log_level = logging.INFO
    if str(os.getenv("DEBUG", "0")).lower() in ("1", "true", "t", "yes", "y"):
        _log_level = logging.DEBUG

    logging.basicConfig(
        stream=sys.stdout,
        level=_log_level,
        datefmt="%Y-%m-%d %H:%M:%S %z",
        format="[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d]: %(message)s",
    )

    _my_class_config = MyClassCliConfig()  # type: ignore
    _my_object = MyClass(items=_my_class_config.items, config=_my_class_config)
    _items = _my_object.run()
    logger.info(f"Items: {_items}")

    return


if __name__ == "__main__":
    main()
