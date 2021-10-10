# read version from installed package
from importlib.metadata import version
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', 
                    datefmt='%y-%m-%d %H:%M')
__version__ = version("instrumentum")

