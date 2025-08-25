"""Type variable system for the DK worklist algorithm."""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass

from ..core import CoreType

# Type aliases
TyVar = str
TmVar = str


class TyVarKind(ABC):
    """Kinds of type variables in the worklist."""
    pass


@dataclass
class UniversalTyVar(TyVarKind):
    """Universal type variable: alpha"""
    pass


@dataclass
class ExistentialTyVar(TyVarKind):
    """Existential type variable: ^alpha"""
    pass


@dataclass
class SolvedTyVar(TyVarKind):
    """Solved existential: ^alpha = tau"""
    solution: CoreType


@dataclass
class MarkerTyVar(TyVarKind):
    """Marker: >alpha (for scoping)"""
    pass