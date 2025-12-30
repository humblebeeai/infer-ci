#!/usr/bin/env python

# Standard libraries
import sys
import logging

# Internal modules
from infer-ci import MyClass


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S %z",
        format="[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d]: %(message)s",
    )

    # Pre-defined variables (for customizing and testing)
    _items = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _config = {
        "min_length": 4,
        "max_length": 10,
        "min_value": 0.0,
        "max_value": 1.0,
        "threshold": 0.7,
    }

    # Main example code
    logger.info(f"Items before cleaning: {_items}")
    _my_object = MyClass(items=_items, config=_config)
    _items = _my_object.run()
    logger.info(f"Items after cleaning: {_items}")

    logger.info("Done!\n")
    return


if __name__ == "__main__":
    main()
