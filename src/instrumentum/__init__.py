# read version from installed package
import logging
from importlib.metadata import version

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', 
                    datefmt='%y-%m-%d %H:%M')

__version__ = version("instrumentum")

