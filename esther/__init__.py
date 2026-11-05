"""esther — economic MPC for house heating over a self-learning grey-box model.

See docs/spec.md. v1 is under construction; M0 provides the package skeleton
and the house loader.
"""

from .house import House, HouseError, load_house, loads_house

__version__ = "1.0.0.dev0"

__all__ = ["House", "HouseError", "__version__", "load_house", "loads_house"]
