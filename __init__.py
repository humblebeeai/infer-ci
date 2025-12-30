# flake8: noqa

try:
    from .src.infer-ci import *
except ImportError:
    from src.infer-ci import *
