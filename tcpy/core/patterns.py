"""Pattern matching for System-F-omega."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class CorePattern(ABC):
    """Core patterns for case expressions."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class WildcardPattern(CorePattern):
    """Wildcard: _"""
    
    def __str__(self) -> str:
        return "_"


@dataclass
class VarPattern(CorePattern):
    """Variable: x"""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass
class ConstructorPattern(CorePattern):
    """Constructor: C p1 ... pn"""
    name: str
    args: List[CorePattern]
    
    def __str__(self) -> str:
        if not self.args:
            return self.name
        return self.name + " " + " ".join(str(arg) for arg in self.args)