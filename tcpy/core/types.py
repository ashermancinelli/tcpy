"""Type system for System-F-omega."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class CoreType(ABC):
    """Core types (System-F-omega)."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class VarType(CoreType):
    """Type variable: a"""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass
class ETVarType(CoreType):
    """Existential type variable: ^a"""
    name: str
    
    def __str__(self) -> str:
        return f"^{self.name}"


@dataclass
class ConType(CoreType):
    """Type constructor: Int, Bool"""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass
class ArrowType(CoreType):
    """Function type: T1 -> T2"""
    t1: CoreType
    t2: CoreType
    
    def __str__(self) -> str:
        return f"{self.t1} -> {self.t2}"


@dataclass
class ForallType(CoreType):
    """Universal quantification: forall a. T"""
    var: str
    ty: CoreType
    
    def __str__(self) -> str:
        return f"forall {self.var}. {self.ty}"


@dataclass
class AppType(CoreType):
    """Type application: F T"""
    t1: CoreType
    t2: CoreType
    
    def __str__(self) -> str:
        return f"({self.t1} {self.t2})"


@dataclass
class ProductType(CoreType):
    """Product type: T1 * T2"""
    types: List[CoreType]
    
    def __str__(self) -> str:
        if not self.types:
            return "()"
        return "(" + " * ".join(str(t) for t in self.types) + ")"