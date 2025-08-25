"""Kind system for System-F-omega types."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Kind(ABC):
    """Kinds for types."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class StarKind(Kind):
    """Base kind: *"""
    
    def __str__(self) -> str:
        return "*"


@dataclass
class ArrowKind(Kind):
    """Function kind: k1 -> k2"""
    k1: Kind
    k2: Kind
    
    def __str__(self) -> str:
        return f"{self.k1} -> {self.k2}"