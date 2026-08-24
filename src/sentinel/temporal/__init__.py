"""Tier 3 -- temporal logic: confirmation, association, dwell, lifecycle.

No GPU, no video, no models. Unit-testable with synthetic tracks.
"""

from .association import BagState, BaggageAssociator, BagRecord
from .confirm import Candidate, ConfirmRule, TemporalConfirmer
from .state_machine import TemporalStateMachine

__all__ = [
    "BagRecord",
    "BagState",
    "BaggageAssociator",
    "Candidate",
    "ConfirmRule",
    "TemporalConfirmer",
    "TemporalStateMachine",
]
