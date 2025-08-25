"""Type judgment system for the DK worklist algorithm."""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass

from ..core import CoreType, CoreTerm


class Judgment(ABC):
    """Type judgments in the worklist."""
    pass


@dataclass
class SubJudgment(Judgment):
    """Subtyping: A <: B"""
    left: CoreType
    right: CoreType


@dataclass
class InfJudgment(Judgment):
    """Type inference: e |- A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class ChkJudgment(Judgment):
    """Type checking: e <= A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class InfAppJudgment(Judgment):
    """Application inference helper: A * e |- C"""
    func_ty: CoreType
    arg: CoreTerm
    result_ty: CoreType