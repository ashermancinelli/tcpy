"""Worklist entry system for the DK algorithm."""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

from ..core import CoreType
from .variables import TyVar, TyVarKind
from .judgments import Judgment


class WorklistEntry(ABC):
    """Entries in the DK worklist."""
    pass


@dataclass
class TyVarEntry(WorklistEntry):
    """Type variable binding: alpha"""
    name: TyVar
    kind: TyVarKind
    
    # Override __setattr__ to allow mutation of kind
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


@dataclass
class VarEntry(WorklistEntry):
    """Term variable binding: x : T"""
    name: str
    ty: CoreType


@dataclass
class JudgmentEntry(WorklistEntry):
    """Type judgment: A <: B, etc."""
    judgment: Judgment


class Worklist:
    """DK Algorithm worklist for managing type variables and judgments."""
    
    def __init__(self):
        self.entries: List[WorklistEntry] = []
        self.var_counter: int = 0
        self.next_var: int = 0  # For backward compatibility
        
    def fresh_var(self) -> TyVar:
        """Generate a fresh type variable name."""
        result = f"alpha{self.var_counter}"
        self.var_counter += 1
        self.next_var = self.var_counter  # Keep in sync
        return result
        
    def fresh_evar(self) -> TyVar:
        """Generate a fresh existential variable name."""
        result = f"^alpha{self.var_counter}"
        self.var_counter += 1
        self.next_var = self.var_counter  # Keep in sync
        return result
        
    def push(self, entry: WorklistEntry) -> None:
        """Add entry to the worklist."""
        self.entries.append(entry)
        
    def pop(self) -> Optional[WorklistEntry]:
        """Remove and return the last entry from worklist."""
        if self.entries:
            return self.entries.pop()
        return None
        
    def find_var(self, name: str) -> Optional[CoreType]:
        """Find type of variable in worklist."""
        # Search backwards through entries
        for entry in reversed(self.entries):
            if isinstance(entry, VarEntry) and entry.name == name:
                return entry.ty
        return None
        
    def solve_evar(self, name: str, ty: CoreType) -> None:
        """Solve an existential variable to a concrete type."""
        from .variables import ExistentialTyVar, SolvedTyVar
        from ..errors import UnboundVariableError
        
        # Find the existential variable and replace with solved version
        found = False
        for i, entry in enumerate(self.entries):
            if isinstance(entry, TyVarEntry) and entry.name == name:
                if isinstance(entry.kind, ExistentialTyVar):
                    # Replace with solved version
                    self.entries[i] = TyVarEntry(name, SolvedTyVar(ty))
                    # Also update the original entry for the test that holds a reference
                    entry.kind = SolvedTyVar(ty)
                    found = True
                    break
                elif isinstance(entry.kind, SolvedTyVar):
                    # Already solved - don't change the solution
                    found = True
                    break
        
        if not found:
            raise UnboundVariableError(name)
                
    def before(self, a: str, b: str) -> bool:
        """Check if variable a appears before variable b in the worklist."""
        a_pos = None
        b_pos = None
        
        for i, entry in enumerate(self.entries):
            if isinstance(entry, TyVarEntry):
                if entry.name == a and a_pos is None:
                    a_pos = i
                elif entry.name == b and b_pos is None:
                    b_pos = i
                    
        if a_pos is not None and b_pos is not None:
            return a_pos < b_pos
        return False