# read version from installed package
from importlib.metadata import version
import logging

logging.basicConfig(level = logging.INFO)
__version__ = version("instrumentum")

