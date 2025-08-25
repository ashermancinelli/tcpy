"""Core System-F-omega language definitions organized into logical submodules."""

# Re-export all classes for backward compatibility
from .kinds import Kind, StarKind, ArrowKind
from .types import (
    CoreType, VarType, ETVarType, ConType, ArrowType, 
    ForallType, AppType, ProductType
)
from .patterns import CorePattern, WildcardPattern, VarPattern, ConstructorPattern
from .operations import CoreBinOp, AddOp, SubOp, MulOp, DivOp, LtOp, LeOp
from .terms import (
    CoreTerm, VarTerm, LitIntTerm, LambdaTerm, AppTerm, TypeLambdaTerm,
    ConstructorTerm, CaseArm, CaseTerm, BinOpTerm, IfTerm
)
from .module import CoreModule, TypeDef, DataConstructor, TermDef

# For convenience, maintain the same interface as the original core.py
__all__ = [
    # Kinds
    'Kind', 'StarKind', 'ArrowKind',
    
    # Types
    'CoreType', 'VarType', 'ETVarType', 'ConType', 'ArrowType', 
    'ForallType', 'AppType', 'ProductType',
    
    # Patterns
    'CorePattern', 'WildcardPattern', 'VarPattern', 'ConstructorPattern',
    
    # Operations
    'CoreBinOp', 'AddOp', 'SubOp', 'MulOp', 'DivOp', 'LtOp', 'LeOp',
    
    # Terms
    'CoreTerm', 'VarTerm', 'LitIntTerm', 'LambdaTerm', 'AppTerm', 'TypeLambdaTerm',
    'ConstructorTerm', 'CaseArm', 'CaseTerm', 'BinOpTerm', 'IfTerm',
    
    # Module structures
    'CoreModule', 'TypeDef', 'DataConstructor', 'TermDef',
]