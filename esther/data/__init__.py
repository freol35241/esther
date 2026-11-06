"""Dataset building, resampling and gap handling (spec §6.1).

:func:`parse_dataset_config` reads a house's ``[dataset]`` table,
:func:`read_signals` fetches histories from any :class:`HistorySource`, and
:func:`build_dataset` resamples them onto the identification grid with masks,
exclusions and gap segments.
"""

from .config import DatasetConfig, SignalSpec, SpikeRule, Window, parse_dataset_config
from .dataset import Dataset, HistorySource, Segment, build_dataset, read_signals
from .resample import MASK_RULES

__all__ = [
    "MASK_RULES",
    "Dataset",
    "DatasetConfig",
    "HistorySource",
    "Segment",
    "SignalSpec",
    "SpikeRule",
    "Window",
    "build_dataset",
    "parse_dataset_config",
    "read_signals",
]
