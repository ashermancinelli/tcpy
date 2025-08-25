"""DK Worklist Algorithm for System-F-omega type inference, organized into logical submodules."""

# Re-export all classes for backward compatibility
from .variables import TyVar, TmVar, TyVarKind, UniversalTyVar, ExistentialTyVar, SolvedTyVar, MarkerTyVar
from .judgments import Judgment, SubJudgment, InfJudgment, ChkJudgment, InfAppJudgment
from .entries import WorklistEntry, TyVarEntry, VarEntry, JudgmentEntry, Worklist
from .inference import DKInference

# For convenience, maintain the same interface as the original worklist.py
__all__ = [
    # Type system
    'TyVar', 'TmVar',
    
    # Variables
    'TyVarKind', 'UniversalTyVar', 'ExistentialTyVar', 'SolvedTyVar', 'MarkerTyVar',
    
    # Judgments
    'Judgment', 'SubJudgment', 'InfJudgment', 'ChkJudgment', 'InfAppJudgment',
    
    # Worklist entries
    'WorklistEntry', 'TyVarEntry', 'VarEntry', 'JudgmentEntry', 'Worklist',
    
    # Main inference engine
    'DKInference',
]