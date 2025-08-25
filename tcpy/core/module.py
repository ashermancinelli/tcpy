"""Module and definition structures for System-F-omega."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .kinds import Kind
from .types import CoreType
from .terms import CoreTerm


@dataclass
class CoreModule:
    """Module containing type and term definitions."""
    type_defs: List[TypeDef]
    term_defs: List[TermDef]


@dataclass
class TypeDef:
    """Type definition with constructors."""
    name: str
    kind: Kind
    constructors: List[DataConstructor]


@dataclass
class DataConstructor:
    """Data constructor with type."""
    name: str
    ty: CoreType


@dataclass
class TermDef:
    """Term definition with type and body."""
    name: str
    ty: CoreType
    body: CoreTerm