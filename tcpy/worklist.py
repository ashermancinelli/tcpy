"""DK Worklist Algorithm for System-F-ω type inference."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .core import (CoreType, CoreTerm, CorePattern, CoreBinOp, 
                   ConType, VarType, ETVarType, ArrowType, ForallType, AppType,
                   VarTerm, LitIntTerm, LambdaTerm, AppTerm, TypeLambdaTerm, 
                   ConstructorTerm, BinOpTerm, IfTerm, CaseTerm, AddOp, SubOp, MulOp, DivOp, LtOp, LeOp)
from .errors import TypeResult, Ok, Err, TypeError as TCPyTypeError
from .errors import (
    UnboundVariableError, UnboundDataConstructorError, NotAFunctionError,
    ArityMismatchError, SubtypingError, InstantiationError
)

# Type aliases
TyVar = str
TmVar = str


class TyVarKind(ABC):
    """Kinds of type variables in the worklist."""
    pass


@dataclass
class UniversalTyVar(TyVarKind):
    """Universal type variable: α"""
    pass


@dataclass
class ExistentialTyVar(TyVarKind):
    """Existential type variable: ^α"""
    pass


@dataclass
class SolvedTyVar(TyVarKind):
    """Solved existential: ^α = τ"""
    solution: CoreType


@dataclass
class MarkerTyVar(TyVarKind):
    """Marker: ►α (for scoping)"""
    pass


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
    """Type inference: e ⊢ A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class ChkJudgment(Judgment):
    """Type checking: e ⇐ A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class InfAppJudgment(Judgment):
    """Application inference helper: A • e ⊢ C"""
    func_ty: CoreType
    arg: CoreTerm
    result_ty: CoreType


class WorklistEntry(ABC):
    """Entries in the DK worklist."""
    pass


@dataclass
class TyVarEntry(WorklistEntry):
    """Type variable binding: α"""
    name: TyVar
    kind: TyVarKind


@dataclass
class VarEntry(WorklistEntry):
    """Term variable binding: x : T"""
    name: TmVar
    ty: CoreType


@dataclass
class JudgmentEntry(WorklistEntry):
    """Judgment: Sub A B | Inf e ⊢ A | Chk e ⇐ A"""
    judgment: Judgment


class Worklist:
    """The DK algorithm worklist for type inference."""
    
    def __init__(self):
        self.entries: List[WorklistEntry] = []
        self.next_var: int = 0
    
    def fresh_var(self) -> TyVar:
        """Generate a fresh universal type variable."""
        var = f"α{self.next_var}"
        self.next_var += 1
        return var
    
    def fresh_evar(self) -> TyVar:
        """Generate a fresh existential type variable."""
        var = f"^α{self.next_var}"
        self.next_var += 1
        return var
    
    def push(self, entry: WorklistEntry) -> None:
        """Add an entry to the end of the worklist."""
        self.entries.append(entry)
    
    def pop(self) -> Optional[WorklistEntry]:
        """Remove and return the last entry from the worklist."""
        if self.entries:
            return self.entries.pop()
        return None
    
    def find_var(self, name: str) -> Optional[CoreType]:
        """Find a term variable's type in the worklist (search from end)."""
        for entry in reversed(self.entries):
            if isinstance(entry, VarEntry) and entry.name == name:
                return entry.ty
        return None
    
    def solve_evar(self, name: str, ty: CoreType) -> TypeResult[None]:
        """Solve an existential variable with the given type."""
        for entry in self.entries:
            if isinstance(entry, TyVarEntry) and entry.name == name:
                if isinstance(entry.kind, ExistentialTyVar):
                    entry.kind = SolvedTyVar(ty)
                    return Ok(None)
                elif isinstance(entry.kind, SolvedTyVar):
                    # Already solved, that's OK
                    return Ok(None)
                # Skip other kinds (universal, marker)
        
        return Err(UnboundVariableError(name))
    
    def before(self, a: str, b: str) -> bool:
        """Check if type variable 'a' appears before 'b' in the worklist."""
        pos_a = None
        pos_b = None
        
        for i, entry in enumerate(self.entries):
            if isinstance(entry, TyVarEntry):
                if entry.name == a:
                    pos_a = i
                if entry.name == b:
                    pos_b = i
        
        if pos_a is not None and pos_b is not None:
            return pos_a < pos_b
        return False


class DKInference:
    """DK algorithm type inference engine."""
    
    def __init__(self, data_constructors: Optional[Dict[str, CoreType]] = None,
                 var_context: Optional[Dict[str, CoreType]] = None):
        self.worklist = Worklist()
        self.trace: List[str] = []
        self.data_constructors = data_constructors or {}
        self.var_context = var_context or {}
    
    @classmethod
    def with_context(cls, data_constructors: Dict[str, CoreType],
                     var_context: Dict[str, CoreType]) -> 'DKInference':
        """Create inference engine with given constructor and variable contexts."""
        return cls(data_constructors, var_context)
    
    def check_type(self, term: CoreTerm, expected_ty: CoreType) -> TypeResult[None]:
        """Check that a term has the expected type."""
        self.worklist.push(JudgmentEntry(ChkJudgment(term, expected_ty)))
        return self.solve()
    
    def solve(self) -> TypeResult[None]:
        """Process the worklist until empty or error."""
        while True:
            entry = self.worklist.pop()
            if entry is None:
                break
            
            # Process different entry types
            if isinstance(entry, TyVarEntry) or isinstance(entry, VarEntry):
                # Skip variable bindings during processing
                continue
            elif isinstance(entry, JudgmentEntry):
                result = self.solve_judgment(entry.judgment)
                if isinstance(result, Err):
                    return result
        
        return Ok(None)
    
    def solve_judgment(self, judgment: Judgment) -> TypeResult[None]:
        """Solve a specific judgment."""
        if isinstance(judgment, SubJudgment):
            return self.solve_subtype(judgment.left, judgment.right)
        elif isinstance(judgment, InfJudgment):
            return self.solve_inference(judgment.term, judgment.ty)
        elif isinstance(judgment, ChkJudgment):
            return self.solve_checking(judgment.term, judgment.ty)
        elif isinstance(judgment, InfAppJudgment):
            return self.solve_inf_app(judgment.func_ty, judgment.arg, judgment.result_ty)
        else:
            # This should not happen with proper typing
            return Err(TCPyTypeError(f"Unknown judgment type: {type(judgment)}"))
    
    def solve_subtype(self, left: CoreType, right: CoreType) -> TypeResult[None]:
        """Solve subtyping constraint left <: right."""
        self.trace.append(f"Sub {left} <: {right}")

        # Reflexivity
        if left == right:
            return Ok(None)

        # Specific cases based on type constructors
        if isinstance(left, ConType) and isinstance(right, ConType):
            if left.name == right.name:
                return Ok(None)
        
        if isinstance(left, VarType) and isinstance(right, VarType):
            if left.name == right.name:
                return Ok(None)
        
        if isinstance(left, ETVarType) and isinstance(right, ETVarType):
            if left.name == right.name:
                return Ok(None)

        # Function subtyping (contravariant in argument, covariant in result)
        if isinstance(left, ArrowType) and isinstance(right, ArrowType):
            # Add subtyping judgments: right.t1 <: left.t1 and left.t2 <: right.t2
            self.worklist.push(JudgmentEntry(SubJudgment(right.t1, left.t1)))  # contravariant
            self.worklist.push(JudgmentEntry(SubJudgment(left.t2, right.t2)))  # covariant
            return Ok(None)

        # Application subtyping (covariant in both components)
        if isinstance(left, AppType) and isinstance(right, AppType):
            self.worklist.push(JudgmentEntry(SubJudgment(left.t1, right.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(left.t2, right.t2)))
            return Ok(None)

        # Forall right: |- A <: ∀α.B  becomes  α |- A <: B
        if isinstance(right, ForallType):
            fresh_var = self.worklist.fresh_var()
            self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
            substituted_ty = self.substitute_type(right.var, VarType(fresh_var), right.ty)
            self.worklist.push(JudgmentEntry(SubJudgment(left, substituted_ty)))
            return Ok(None)

        # Forall left: |- ∀α.A <: B  becomes  ►^α,^α |- [^α/α]A <: B
        if isinstance(left, ForallType):
            fresh_evar = self.worklist.fresh_evar()
            self.worklist.push(TyVarEntry(fresh_evar, MarkerTyVar()))
            self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
            substituted_ty = self.substitute_type(left.var, ETVarType(fresh_evar), left.ty)
            self.worklist.push(JudgmentEntry(SubJudgment(substituted_ty, right)))
            return Ok(None)

        # Existential variable instantiation
        if isinstance(left, ETVarType):
            if not self.occurs_check(left.name, right):
                return self.instantiate_left(left.name, right)
        
        if isinstance(right, ETVarType):
            if not self.occurs_check(right.name, left):
                return self.instantiate_right(left, right.name)

        # If none of the above cases match, subtyping fails
        return Err(SubtypingError(left, right))
    
    def solve_inference(self, term: CoreTerm, ty: CoreType) -> TypeResult[None]:
        """Solve type inference judgment: term ⊢ ty."""
        self.trace.append(f"Inf {self.term_to_string(term)} ⊢ {ty}")

        if isinstance(term, VarTerm):
            # Check pattern variable context first
            if term.name in self.var_context:
                var_ty = self.var_context[term.name]
                self.worklist.push(JudgmentEntry(SubJudgment(var_ty, ty)))
                return Ok(None)
            
            # Check worklist for variable bindings
            found_ty = self.worklist.find_var(term.name)
            if found_ty:
                self.worklist.push(JudgmentEntry(SubJudgment(found_ty, ty)))
                return Ok(None)
            
            # Check data constructors
            if term.name in self.data_constructors:
                constructor_ty = self.data_constructors[term.name]
                self.worklist.push(JudgmentEntry(SubJudgment(constructor_ty, ty)))
                return Ok(None)
            
            return Err(UnboundVariableError(term.name))

        elif isinstance(term, LitIntTerm):
            # Integer literals have type Int
            self.worklist.push(JudgmentEntry(SubJudgment(ConType("Int"), ty)))
            return Ok(None)

        elif isinstance(term, LambdaTerm):
            # λx:T. e  should infer  T -> T'  where e ⊢ T'
            result_ty = ETVarType(self.worklist.fresh_evar())
            self.worklist.push(TyVarEntry(result_ty.name, ExistentialTyVar()))
            
            arrow_ty = ArrowType(term.param_ty, result_ty)
            self.worklist.push(JudgmentEntry(SubJudgment(arrow_ty, ty)))
            
            # Add parameter to variable context and check body
            self.worklist.push(VarEntry(term.param, term.param_ty))
            self.worklist.push(JudgmentEntry(InfJudgment(term.body, result_ty)))
            return Ok(None)

        elif isinstance(term, AppTerm):
            # e1 e2  where  e1 ⊢ T1  and we need T1 • e2 ⊢ ty
            func_ty = ETVarType(self.worklist.fresh_evar())
            self.worklist.push(TyVarEntry(func_ty.name, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(InfAppJudgment(func_ty, term.arg, ty)))
            self.worklist.push(JudgmentEntry(InfJudgment(term.func, func_ty)))
            return Ok(None)

        elif isinstance(term, TypeLambdaTerm):
            # Λα. e  should infer  ∀α. T  where e ⊢ T
            body_ty = ETVarType(self.worklist.fresh_evar())
            self.worklist.push(TyVarEntry(body_ty.name, ExistentialTyVar()))
            
            forall_ty = ForallType(term.param, body_ty)
            self.worklist.push(JudgmentEntry(SubJudgment(forall_ty, ty)))
            
            # Add type variable to context and check body
            self.worklist.push(TyVarEntry(term.param, UniversalTyVar()))
            self.worklist.push(JudgmentEntry(InfJudgment(term.body, body_ty)))
            return Ok(None)

        elif isinstance(term, ConstructorTerm):
            # Constructor application
            if term.name in self.data_constructors:
                constructor_ty = self.data_constructors[term.name]
                # TODO: Handle constructor arguments properly
                # For now, just check the constructor type matches
                self.worklist.push(JudgmentEntry(SubJudgment(constructor_ty, ty)))
                return Ok(None)
            else:
                return Err(UnboundDataConstructorError(term.name))

        elif isinstance(term, BinOpTerm):
            # Binary operations
            left_ty, right_ty, result_ty = self.infer_binop_types(term.op)
            
            # Check operands have correct types
            self.worklist.push(JudgmentEntry(ChkJudgment(term.left, left_ty)))
            self.worklist.push(JudgmentEntry(ChkJudgment(term.right, right_ty)))
            
            # Check result type
            self.worklist.push(JudgmentEntry(SubJudgment(result_ty, ty)))
            return Ok(None)

        elif isinstance(term, IfTerm):
            # if e1 then e2 else e3
            # e1 must be Bool, e2 and e3 must have type ty
            self.worklist.push(JudgmentEntry(ChkJudgment(term.cond, ConType("Bool"))))
            self.worklist.push(JudgmentEntry(ChkJudgment(term.then_branch, ty)))
            self.worklist.push(JudgmentEntry(ChkJudgment(term.else_branch, ty)))
            return Ok(None)

        elif isinstance(term, CaseTerm):
            # case e of { p1 -> e1; ... }
            # Create fresh type for scrutinee
            scrutinee_ty = ETVarType(self.worklist.fresh_evar())
            self.worklist.push(TyVarEntry(scrutinee_ty.name, ExistentialTyVar()))
            
            # Check scrutinee
            self.worklist.push(JudgmentEntry(ChkJudgment(term.scrutinee, scrutinee_ty)))
            
            # Check each arm (simplified - full pattern matching would be more complex)
            for arm in term.arms:
                # For now, just check the body has the right type
                # TODO: Handle pattern variable bindings properly
                self.worklist.push(JudgmentEntry(ChkJudgment(arm.body, ty)))
            
            return Ok(None)

        else:
            return Err(TCPyTypeError(f"Unsupported term for inference: {type(term)}"))
    
    def solve_checking(self, term: CoreTerm, ty: CoreType) -> TypeResult[None]:
        """Solve type checking judgment: term ⇐ ty."""
        self.trace.append(f"Chk {self.term_to_string(term)} ⇐ {ty}")

        # Special case: lambda against arrow type
        if isinstance(term, LambdaTerm) and isinstance(ty, ArrowType):
            # λx:T₁. e ⇐ T₁ -> T₂  becomes  T₁ <: T₁, x:T₁ ⊢ e ⇐ T₂
            self.worklist.push(JudgmentEntry(SubJudgment(term.param_ty, ty.t1)))
            self.worklist.push(VarEntry(term.param, term.param_ty))
            self.worklist.push(JudgmentEntry(ChkJudgment(term.body, ty.t2)))
            return Ok(None)

        # Special case: checking against forall type
        if isinstance(ty, ForallType):
            # e ⇐ ∀α.A  becomes  α ⊢ e ⇐ A
            fresh_var = self.worklist.fresh_var()
            self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
            substituted_ty = self.substitute_type(ty.var, VarType(fresh_var), ty.ty)
            self.worklist.push(JudgmentEntry(ChkJudgment(term, substituted_ty)))
            return Ok(None)

        # General case: infer type and check subtyping
        # e ⇐ A  becomes  e ⊢ ^α, ^α <: A
        inferred_ty = ETVarType(self.worklist.fresh_evar())
        self.worklist.push(TyVarEntry(inferred_ty.name, ExistentialTyVar()))
        
        self.worklist.push(JudgmentEntry(SubJudgment(inferred_ty, ty)))
        self.worklist.push(JudgmentEntry(InfJudgment(term, inferred_ty)))
        return Ok(None)
    
    def solve_inf_app(self, func_ty: CoreType, arg: CoreTerm, result_ty: CoreType) -> TypeResult[None]:
        """Solve application inference judgment: func_ty • arg ⊢ result_ty."""
        self.trace.append(f"InfApp {func_ty} • {self.term_to_string(arg)} ⊢ {result_ty}")

        if isinstance(func_ty, ArrowType):
            # T1 -> T2 • e ⊢ A  becomes  T2 <: A, e ⇐ T1
            self.worklist.push(JudgmentEntry(SubJudgment(func_ty.t2, result_ty)))
            self.worklist.push(JudgmentEntry(ChkJudgment(arg, func_ty.t1)))
            return Ok(None)

        elif isinstance(func_ty, ForallType):
            # ∀α.A • e ⊢ C  becomes  [^β/α]A • e ⊢ C  where ^β is fresh
            fresh_evar = self.worklist.fresh_evar()
            self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
            substituted = self.substitute_type(func_ty.var, ETVarType(fresh_evar), func_ty.ty)
            self.worklist.push(JudgmentEntry(InfAppJudgment(substituted, arg, result_ty)))
            return Ok(None)

        elif isinstance(func_ty, ETVarType):
            # ^α • e ⊢ A  becomes  ^α = ^β -> ^γ, ^γ <: A, e ⇐ ^β
            param_ty_name = self.worklist.fresh_evar()
            ret_ty_name = self.worklist.fresh_evar()
            param_ty = ETVarType(param_ty_name)
            ret_ty = ETVarType(ret_ty_name)

            # Add fresh variables to worklist
            self.worklist.push(TyVarEntry(param_ty_name, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(ret_ty_name, ExistentialTyVar()))

            # Solve existential variable
            arrow_ty = ArrowType(param_ty, ret_ty)
            result = self.worklist.solve_evar(func_ty.name, arrow_ty)
            if isinstance(result, Err):
                return result

            # Add subtyping and checking judgments
            self.worklist.push(JudgmentEntry(SubJudgment(ret_ty, result_ty)))
            self.worklist.push(JudgmentEntry(ChkJudgment(arg, param_ty)))
            return Ok(None)

        else:
            return Err(NotAFunctionError(func_ty))
    
    def term_to_string(self, term: CoreTerm) -> str:
        """Convert term to string for tracing."""
        # Use the existing __str__ methods from core.py
        return str(term)
    
    def get_trace(self) -> List[str]:
        """Get the inference trace for debugging."""
        return self.trace.copy()
    
    # Helper methods for subtyping and type checking
    
    def occurs_check(self, var: str, ty: CoreType) -> bool:
        """Check if a type variable occurs in a type (to prevent infinite types)."""
        if isinstance(ty, ETVarType) or isinstance(ty, VarType):
            return ty.name == var
        elif isinstance(ty, ArrowType):
            return self.occurs_check(var, ty.t1) or self.occurs_check(var, ty.t2)
        elif isinstance(ty, AppType):
            return self.occurs_check(var, ty.t1) or self.occurs_check(var, ty.t2)
        elif isinstance(ty, ForallType):
            return self.occurs_check(var, ty.ty)
        # For other types like ConType, ProductType, etc., the variable doesn't occur
        return False
    
    def substitute_type(self, var: str, replacement: CoreType, ty: CoreType) -> CoreType:
        """Substitute a type variable with a replacement type."""
        if isinstance(ty, VarType) and ty.name == var:
            return replacement
        elif isinstance(ty, ArrowType):
            return ArrowType(
                self.substitute_type(var, replacement, ty.t1),
                self.substitute_type(var, replacement, ty.t2)
            )
        elif isinstance(ty, ForallType) and ty.var != var:
            # Don't substitute under bindings of the same variable
            return ForallType(
                ty.var,
                self.substitute_type(var, replacement, ty.ty)
            )
        elif isinstance(ty, AppType):
            return AppType(
                self.substitute_type(var, replacement, ty.t1),
                self.substitute_type(var, replacement, ty.t2)
            )
        # For other types, return unchanged
        return ty
    
    def instantiate_left(self, var: str, ty: CoreType) -> TypeResult[None]:
        """Instantiate left existential variable: ^α <: A."""
        if isinstance(ty, ETVarType) and self.worklist.before(var, ty.name):
            # ^α <: ^β where α appears before β
            result = self.worklist.solve_evar(ty.name, ETVarType(var))
            return result
        elif isinstance(ty, ArrowType):
            # ^α <: A -> B  becomes  ^α = ^α1 -> ^α2, A <: ^α1, ^α2 <: B
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            arrow_ty = ArrowType(ETVarType(a1), ETVarType(a2))
            
            result = self.worklist.solve_evar(var, arrow_ty)
            if isinstance(result, Err):
                return result
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            # Note: contravariant in first argument, covariant in second
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(a1))))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a2), ty.t2)))
            return Ok(None)
        elif isinstance(ty, AppType):
            # ^α <: F A  becomes  ^α = ^α1 ^α2, ^α1 <: F, ^α2 <: A
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            app_ty = AppType(ETVarType(a1), ETVarType(a2))
            
            result = self.worklist.solve_evar(var, app_ty)
            if isinstance(result, Err):
                return result
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a1), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a2), ty.t2)))
            return Ok(None)
        elif isinstance(ty, ForallType):
            # ^α <: ∀β.B  becomes  ►β,β |- ^α <: B
            fresh_var = self.worklist.fresh_var()
            self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
            substituted = self.substitute_type(ty.var, VarType(fresh_var), ty.ty)
            return self.instantiate_left(var, substituted)
        elif self.is_monotype(ty):
            # For monotypes, just solve directly
            return self.worklist.solve_evar(var, ty)
        else:
            return Err(InstantiationError(var, ty))
    
    def instantiate_right(self, ty: CoreType, var: str) -> TypeResult[None]:
        """Instantiate right existential variable: A <: ^α."""
        if isinstance(ty, ETVarType) and self.worklist.before(var, ty.name):
            # ^β <: ^α where α appears before β
            result = self.worklist.solve_evar(ty.name, ETVarType(var))
            return result
        elif isinstance(ty, ArrowType):
            # A -> B <: ^α  becomes  ^α = ^α1 -> ^α2, ^α1 <: A, B <: ^α2
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            arrow_ty = ArrowType(ETVarType(a1), ETVarType(a2))
            
            result = self.worklist.solve_evar(var, arrow_ty)
            if isinstance(result, Err):
                return result
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a1), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(a2))))
            return Ok(None)
        elif isinstance(ty, AppType):
            # F A <: ^α  becomes  ^α = ^α1 ^α2, F <: ^α1, A <: ^α2
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            app_ty = AppType(ETVarType(a1), ETVarType(a2))
            
            result = self.worklist.solve_evar(var, app_ty)
            if isinstance(result, Err):
                return result
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(a1))))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(a2))))
            return Ok(None)
        elif isinstance(ty, ForallType):
            # ∀β.B <: ^α  becomes  ►^β,^β |- [^β/β]B <: ^α
            fresh_evar = self.worklist.fresh_evar()
            self.worklist.push(TyVarEntry(fresh_evar, MarkerTyVar()))
            self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
            substituted = self.substitute_type(ty.var, ETVarType(fresh_evar), ty.ty)
            return self.instantiate_right(substituted, var)
        elif self.is_monotype(ty):
            # For monotypes, just solve directly
            return self.worklist.solve_evar(var, ty)
        else:
            return Err(InstantiationError(var, ty))
    
    def is_monotype(self, ty: CoreType) -> bool:
        """Check if a type is a monotype (no forall quantifiers)."""
        if isinstance(ty, (ConType, VarType, ETVarType)):
            return True
        elif isinstance(ty, (ArrowType, AppType)):
            return self.is_monotype(ty.t1) and self.is_monotype(ty.t2)
        elif isinstance(ty, ForallType):
            return False
        # For other types like ProductType, assume they're monotypes for now
        return True
    
    def infer_binop_types(self, op: CoreBinOp) -> Tuple[CoreType, CoreType, CoreType]:
        """Infer types for binary operations: (left_type, right_type, result_type)."""
        if isinstance(op, (AddOp, SubOp, MulOp, DivOp)):
            # Arithmetic operations: Int -> Int -> Int
            return (ConType("Int"), ConType("Int"), ConType("Int"))
        elif isinstance(op, (LtOp, LeOp)):
            # Comparison operations: Int -> Int -> Bool
            return (ConType("Int"), ConType("Int"), ConType("Bool"))
        else:
            # Default fallback - shouldn't happen with proper typing
            return (ConType("Int"), ConType("Int"), ConType("Int"))