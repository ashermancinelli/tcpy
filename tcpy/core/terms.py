"""Term system for System-F-omega."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .types import CoreType
from .patterns import CorePattern
from .operations import CoreBinOp


class CoreTerm(ABC):
    """Core terms (System-F)."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class VarTerm(CoreTerm):
    """Variable: x"""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass
class LitIntTerm(CoreTerm):
    """Integer literal: 42"""
    value: int
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class LambdaTerm(CoreTerm):
    """Lambda abstraction: lambda x:T. e"""
    param: str
    param_ty: CoreType
    body: CoreTerm
    
    def __str__(self) -> str:
        return f"lambda {self.param} : {self.param_ty}. {self.body}"


@dataclass
class AppTerm(CoreTerm):
    """Application: e1 e2"""
    func: CoreTerm
    arg: CoreTerm
    
    def __str__(self) -> str:
        return f"{self.func} {self.arg}"


@dataclass
class TypeLambdaTerm(CoreTerm):
    """Type abstraction: Lambda alpha. e"""
    param: str
    body: CoreTerm
    
    def __str__(self) -> str:
        return f"Lambda {self.param}. {self.body}"


@dataclass
class ConstructorTerm(CoreTerm):
    """Constructor: C e1 ... en"""
    name: str
    args: List[CoreTerm]
    
    def __str__(self) -> str:
        if not self.args:
            return self.name
        return self.name + " " + " ".join(str(arg) for arg in self.args)


@dataclass
class CaseArm:
    """Case arm with pattern and body."""
    pattern: CorePattern
    body: CoreTerm


@dataclass
class CaseTerm(CoreTerm):
    """Pattern matching: case e of { p1 -> e1; ... }"""
    scrutinee: CoreTerm
    arms: List[CaseArm]
    
    def __str__(self) -> str:
        return f"match {self.scrutinee} {{ ... }}"


@dataclass
class BinOpTerm(CoreTerm):
    """Built-in operations"""
    op: CoreBinOp
    left: CoreTerm
    right: CoreTerm
    
    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass
class IfTerm(CoreTerm):
    """Conditional: if e1 then e2 else e3"""
    cond: CoreTerm
    then_branch: CoreTerm
    else_branch: CoreTerm
    
    def __str__(self) -> str:
        return f"if {self.cond} then {self.then_branch} else {self.else_branch}"