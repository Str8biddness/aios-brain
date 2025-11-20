"""Reasoning module: Deductive, Inductive, and Abductive reasoning engines.

This module provides multiple reasoning paradigms for the AIOS Brain:
- Deductive: Logical inference from known rules
- Inductive: Pattern recognition and generalization
- Abductive: Best explanation hypothesis generation
"""

from .deductive import DeductiveEngine
from .inductive import InductiveEngine
from .abductive import AbductiveEngine, Hypothesis
from .context_manager import ContextManager

__all__ = [
    'DeductiveEngine',
    'InductiveEngine',
    'AbductiveEngine',
    'Hypothesis',
    'ContextManager',
]

__version__ = '0.1.0'
__author__ = 'AIOS Team'
