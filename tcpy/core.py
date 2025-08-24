"""Core language for System-F-omega with explicit type applications and abstractions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union
from abc import ABC, abstractmethod


@dataclass
class CoreModule:
    """Module containing type and term definitions."""
    type_defs: List[TypeDef]
    term_defs: List[TermDef]


@dataclass
class TypeDef:
    """Type definition with constructors."""
    name: str
    kind: 'Kind'
    constructors: List[DataConstructor]


@dataclass
class DataConstructor:
    """Data constructor with type."""
    name: str
    ty: 'CoreType'


@dataclass
class TermDef:
    """Term definition with type and body."""
    name: str
    ty: 'CoreType'
    body: 'CoreTerm'


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
    pattern: 'CorePattern'
    body: CoreTerm


@dataclass
class CaseTerm(CoreTerm):
    """Pattern matching: case e of { p1 -> e1; ... }"""
    scrutinee: CoreTerm
    arms: List[CaseArm]
    
    def __str__(self) -> str:
        return f"match {self.scrutinee} {{ ... }}"


class CoreBinOp(ABC):
    """Binary operations."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class AddOp(CoreBinOp):
    def __str__(self) -> str:
        return "+"


@dataclass
class SubOp(CoreBinOp):
    def __str__(self) -> str:
        return "-"


@dataclass
class MulOp(CoreBinOp):
    def __str__(self) -> str:
        return "*"


@dataclass
class DivOp(CoreBinOp):
    def __str__(self) -> str:
        return "/"


@dataclass
class LtOp(CoreBinOp):
    def __str__(self) -> str:
        return "<"


@dataclass
class LeOp(CoreBinOp):
    def __str__(self) -> str:
        return "<="


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