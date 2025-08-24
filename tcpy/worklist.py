"""DK Worklist Algorithm for System-F-omega type inference."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .core import (CoreType, CoreTerm, CorePattern, CoreBinOp, 
                   ConType, VarType, ETVarType, ArrowType, ForallType, AppType,
                   VarTerm, LitIntTerm, LambdaTerm, AppTerm, TypeLambdaTerm, 
                   ConstructorTerm, BinOpTerm, IfTerm, CaseTerm, AddOp, SubOp, MulOp, DivOp, LtOp, LeOp)
from .errors import TypeError as TCPyTypeError
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
    """Universal type variable: alpha"""
    pass


@dataclass
class ExistentialTyVar(TyVarKind):
    """Existential type variable: ^alpha"""
    pass


@dataclass
class SolvedTyVar(TyVarKind):
    """Solved existential: ^alpha = tau"""
    solution: CoreType


@dataclass
class MarkerTyVar(TyVarKind):
    """Marker: >alpha (for scoping)"""
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
    """Type inference: e |- A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class ChkJudgment(Judgment):
    """Type checking: e <= A"""
    term: CoreTerm
    ty: CoreType


@dataclass
class InfAppJudgment(Judgment):
    """Application inference helper: A * e |- C"""
    func_ty: CoreType
    arg: CoreTerm
    result_ty: CoreType


class WorklistEntry(ABC):
    """Entries in the DK worklist."""
    pass


@dataclass
class TyVarEntry(WorklistEntry):
    """Type variable binding: alpha"""
    name: TyVar
    kind: TyVarKind


@dataclass
class VarEntry(WorklistEntry):
    """Term variable binding: x : T"""
    name: TmVar
    ty: CoreType


@dataclass
class JudgmentEntry(WorklistEntry):
    """Judgment: Sub A B | Inf e |- A | Chk e <= A"""
    judgment: Judgment


class Worklist:
    """The DK algorithm worklist for type inference."""
    
    def __init__(self):
        self.entries: List[WorklistEntry] = []
        self.next_var: int = 0
    
    def fresh_var(self) -> TyVar:
        """Generate a fresh universal type variable."""
        var = f"alpha{self.next_var}"
        self.next_var += 1
        return var
    
    def fresh_evar(self) -> TyVar:
        """Generate a fresh existential type variable."""
        var = f"^alpha{self.next_var}"
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
            match entry:
                case VarEntry(entry_name, ty) if entry_name == name:
                    return ty
        return None
    
    def solve_evar(self, name: str, ty: CoreType) -> None:
        """Solve an existential variable with the given type."""
        for entry in self.entries:
            match entry:
                case TyVarEntry(entry_name, kind) if entry_name == name:
                    match kind:
                        case ExistentialTyVar():
                            entry.kind = SolvedTyVar(ty)
                            return
                        case SolvedTyVar(_):
                            # Already solved, that's OK
                            return
                        case _:
                            # Skip other kinds (universal, marker)
                            continue
        
        raise UnboundVariableError(name)
    
    def before(self, a: str, b: str) -> bool:
        """Check if type variable 'a' appears before 'b' in the worklist."""
        pos_a = None
        pos_b = None
        
        for i, entry in enumerate(self.entries):
            match entry:
                case TyVarEntry(name, _):
                    if name == a:
                        pos_a = i
                    if name == b:
                        pos_b = i
        
        return pos_a is not None and pos_b is not None and pos_a < pos_b


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
    
    def check_type(self, term: CoreTerm, expected_ty: CoreType) -> None:
        """Check that a term has the expected type."""
        self.worklist.push(JudgmentEntry(ChkJudgment(term, expected_ty)))
        self.solve()
    
    def solve(self) -> None:
        """Process the worklist until empty or error."""
        while True:
            entry = self.worklist.pop()
            if entry is None:
                break
            
            # Process different entry types
            match entry:
                case TyVarEntry(_, _) | VarEntry(_, _):
                    # Skip variable bindings during processing
                    continue
                case JudgmentEntry(judgment):
                    self.solve_judgment(judgment)
    
    def solve_judgment(self, judgment: Judgment) -> None:
        """Solve a specific judgment."""
        match judgment:
            case SubJudgment(left, right):
                self.solve_subtype(left, right)
            case InfJudgment(term, ty):
                self.solve_inference(term, ty)
            case ChkJudgment(term, ty):
                self.solve_checking(term, ty)
            case InfAppJudgment(func_ty, arg, result_ty):
                self.solve_inf_app(func_ty, arg, result_ty)
            case _:
                # This should not happen with proper typing
                raise TCPyTypeError(f"Unknown judgment type: {type(judgment)}")
    
    def solve_subtype(self, left: CoreType, right: CoreType) -> None:
        """Solve subtyping constraint left <: right."""
        self.trace.append(f"Sub {left} <: {right}")

        # Reflexivity
        if left == right:
            return

        # Pattern match on type combinations
        match (left, right):
            # Specific cases based on type constructors
            case (ConType(left_name), ConType(right_name)) if left_name == right_name:
                return
            case (VarType(left_name), VarType(right_name)) if left_name == right_name:
                return
            case (ETVarType(left_name), ETVarType(right_name)) if left_name == right_name:
                return

            # Function subtyping (contravariant in argument, covariant in result)
            case (ArrowType(l_param, l_ret), ArrowType(r_param, r_ret)):
                # Add subtyping judgments: right.t1 <: left.t1 and left.t2 <: right.t2
                self.worklist.push(JudgmentEntry(SubJudgment(r_param, l_param)))  # contravariant
                self.worklist.push(JudgmentEntry(SubJudgment(l_ret, r_ret)))  # covariant
                return

            # Application subtyping (covariant in both components)
            case (AppType(l_func, l_arg), AppType(r_func, r_arg)):
                self.worklist.push(JudgmentEntry(SubJudgment(l_func, r_func)))
                self.worklist.push(JudgmentEntry(SubJudgment(l_arg, r_arg)))
                return

            # Forall right: |- A <: forall alpha.B  becomes  alpha |- A <: B
            case (_, ForallType(var, body)):
                fresh_var = self.worklist.fresh_var()
                self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
                substituted_ty = self.substitute_type(var, VarType(fresh_var), body)
                self.worklist.push(JudgmentEntry(SubJudgment(left, substituted_ty)))
                return

            # Forall left: |- forall alpha.A <: B  becomes  ►^alpha,^alpha |- [^alpha/alpha]A <: B
            case (ForallType(var, body), _):
                fresh_evar = self.worklist.fresh_evar()
                self.worklist.push(TyVarEntry(fresh_evar, MarkerTyVar()))
                self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
                substituted_ty = self.substitute_type(var, ETVarType(fresh_evar), body)
                self.worklist.push(JudgmentEntry(SubJudgment(substituted_ty, right)))
                return

            # Existential variable instantiation
            case (ETVarType(var_name), _) if not self.occurs_check(var_name, right):
                self.instantiate_left(var_name, right)
                return
            case (_, ETVarType(var_name)) if not self.occurs_check(var_name, left):
                self.instantiate_right(left, var_name)
                return

        # If none of the above cases match, subtyping fails
        raise SubtypingError(left, right)
    
    def solve_inference(self, term: CoreTerm, ty: CoreType) -> None:
        """Solve type inference judgment: term |- ty."""
        self.trace.append(f"Inf {self.term_to_string(term)} |- {ty}")

        match term:
            case VarTerm(name):
                # Check pattern variable context first
                if name in self.var_context:
                    var_ty = self.var_context[name]
                    self.worklist.push(JudgmentEntry(SubJudgment(var_ty, ty)))
                    return
                
                # Check worklist for variable bindings
                found_ty = self.worklist.find_var(name)
                if found_ty:
                    self.worklist.push(JudgmentEntry(SubJudgment(found_ty, ty)))
                    return
                
                # Check data constructors
                if name in self.data_constructors:
                    constructor_ty = self.data_constructors[name]
                    self.worklist.push(JudgmentEntry(SubJudgment(constructor_ty, ty)))
                    return
                
                raise UnboundVariableError(name)

            case LitIntTerm(_):
                # Integer literals have type Int
                self.worklist.push(JudgmentEntry(SubJudgment(ConType("Int"), ty)))
                return

            case LambdaTerm(param, param_ty, body):
                # lambda x:T. e  should infer  T -> T'  where e |- T'
                result_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(result_ty.name, ExistentialTyVar()))
                
                arrow_ty = ArrowType(param_ty, result_ty)
                self.worklist.push(JudgmentEntry(SubJudgment(arrow_ty, ty)))
                
                # Add parameter to variable context and check body
                self.worklist.push(VarEntry(param, param_ty))
                self.worklist.push(JudgmentEntry(InfJudgment(body, result_ty)))
                return

            case AppTerm(func, arg):
                # e1 e2  where  e1 |- T1  and we need T1 * e2 |- ty
                func_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(func_ty.name, ExistentialTyVar()))
                
                self.worklist.push(JudgmentEntry(InfAppJudgment(func_ty, arg, ty)))
                self.worklist.push(JudgmentEntry(InfJudgment(func, func_ty)))
                return

            case TypeLambdaTerm(param, body):
                # Lambda alpha. e  should infer  forall alpha. T  where e |- T
                body_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(body_ty.name, ExistentialTyVar()))
                
                forall_ty = ForallType(param, body_ty)
                self.worklist.push(JudgmentEntry(SubJudgment(forall_ty, ty)))
                
                # Add type variable to context and check body
                self.worklist.push(TyVarEntry(param, UniversalTyVar()))
                self.worklist.push(JudgmentEntry(InfJudgment(body, body_ty)))
                return

            case ConstructorTerm(name, args):
                # Constructor application
                if name in self.data_constructors:
                    constructor_ty = self.data_constructors[name]
                    # TODO: Handle constructor arguments properly
                    # For now, just check the constructor type matches
                    self.worklist.push(JudgmentEntry(SubJudgment(constructor_ty, ty)))
                    return
                else:
                    raise UnboundDataConstructorError(name)

            case BinOpTerm(op, left, right):
                # Binary operations
                left_ty, right_ty, result_ty = self.infer_binop_types(op)
                
                # Check operands have correct types
                self.worklist.push(JudgmentEntry(ChkJudgment(left, left_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(right, right_ty)))
                
                # Check result type
                self.worklist.push(JudgmentEntry(SubJudgment(result_ty, ty)))
                return

            case IfTerm(cond, then_branch, else_branch):
                # if e1 then e2 else e3
                # e1 must be Bool, e2 and e3 must have type ty
                self.worklist.push(JudgmentEntry(ChkJudgment(cond, ConType("Bool"))))
                self.worklist.push(JudgmentEntry(ChkJudgment(then_branch, ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(else_branch, ty)))
                return

            case CaseTerm(scrutinee, arms):
                # case e of { p1 -> e1; ... }
                # Create fresh type for scrutinee
                scrutinee_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(scrutinee_ty.name, ExistentialTyVar()))
                
                # Check scrutinee
                self.worklist.push(JudgmentEntry(ChkJudgment(scrutinee, scrutinee_ty)))
                
                # Check each arm (simplified - full pattern matching would be more complex)
                for arm in arms:
                    # For now, just check the body has the right type
                    # TODO: Handle pattern variable bindings properly
                    self.worklist.push(JudgmentEntry(ChkJudgment(arm.body, ty)))
                
                return

            case _:
                raise TCPyTypeError(f"Unsupported term for inference: {type(term)}")
    
    def solve_checking(self, term: CoreTerm, ty: CoreType) -> None:
        """Solve type checking judgment: term <= ty."""
        self.trace.append(f"Chk {self.term_to_string(term)} <= {ty}")

        match (term, ty):
            # Special case: lambda against arrow type
            case (LambdaTerm(param, param_ty, body), ArrowType(expected_param, result_ty)):
                # lambda x:T₁. e <= T₁ -> T₂  becomes  T₁ <: T₁, x:T₁ |- e <= T₂
                self.worklist.push(JudgmentEntry(SubJudgment(param_ty, expected_param)))
                self.worklist.push(VarEntry(param, param_ty))
                self.worklist.push(JudgmentEntry(ChkJudgment(body, result_ty)))
                return

            # Special case: checking against forall type
            case (_, ForallType(var, body)):
                # e <= forall alpha.A  becomes  alpha |- e <= A
                fresh_var = self.worklist.fresh_var()
                self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
                substituted_ty = self.substitute_type(var, VarType(fresh_var), body)
                self.worklist.push(JudgmentEntry(ChkJudgment(term, substituted_ty)))
                return

            # General case: infer type and check subtyping
            case _:
                # e <= A  becomes  e |- ^alpha, ^alpha <: A
                inferred_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(inferred_ty.name, ExistentialTyVar()))
                
                self.worklist.push(JudgmentEntry(SubJudgment(inferred_ty, ty)))
                self.worklist.push(JudgmentEntry(InfJudgment(term, inferred_ty)))
                return
    
    def solve_inf_app(self, func_ty: CoreType, arg: CoreTerm, result_ty: CoreType) -> None:
        """Solve application inference judgment: func_ty * arg |- result_ty."""
        self.trace.append(f"InfApp {func_ty} * {self.term_to_string(arg)} |- {result_ty}")

        match func_ty:
            case ArrowType(param_ty, ret_ty):
                # T1 -> T2 * e |- A  becomes  T2 <: A, e <= T1
                self.worklist.push(JudgmentEntry(SubJudgment(ret_ty, result_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(arg, param_ty)))
                return

            case ForallType(var, body):
                # forall alpha.A * e |- C  becomes  [^beta/alpha]A * e |- C  where ^beta is fresh
                fresh_evar = self.worklist.fresh_evar()
                self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
                substituted = self.substitute_type(var, ETVarType(fresh_evar), body)
                self.worklist.push(JudgmentEntry(InfAppJudgment(substituted, arg, result_ty)))
                return

            case ETVarType(var_name):
                # ^alpha * e |- A  becomes  ^alpha = ^beta -> ^gamma, ^gamma <: A, e <= ^beta
                param_ty_name = self.worklist.fresh_evar()
                ret_ty_name = self.worklist.fresh_evar()
                param_ty = ETVarType(param_ty_name)
                ret_ty = ETVarType(ret_ty_name)

                # Add fresh variables to worklist
                self.worklist.push(TyVarEntry(param_ty_name, ExistentialTyVar()))
                self.worklist.push(TyVarEntry(ret_ty_name, ExistentialTyVar()))

                # Solve existential variable
                arrow_ty = ArrowType(param_ty, ret_ty)
                self.worklist.solve_evar(var_name, arrow_ty)

                # Add subtyping and checking judgments
                self.worklist.push(JudgmentEntry(SubJudgment(ret_ty, result_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(arg, param_ty)))
                return

            case _:
                raise NotAFunctionError(func_ty)
    
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
        match ty:
            case ETVarType(name) | VarType(name):
                return name == var
            case ArrowType(t1, t2) | AppType(t1, t2):
                return self.occurs_check(var, t1) or self.occurs_check(var, t2)
            case ForallType(_, body):
                return self.occurs_check(var, body)
            case _:
                # For other types like ConType, ProductType, etc., the variable doesn't occur
                return False
    
    def substitute_type(self, var: str, replacement: CoreType, ty: CoreType) -> CoreType:
        """Substitute a type variable with a replacement type."""
        match ty:
            case VarType(name) if name == var:
                return replacement
            case ArrowType(t1, t2):
                return ArrowType(
                    self.substitute_type(var, replacement, t1),
                    self.substitute_type(var, replacement, t2)
                )
            case ForallType(bound_var, body) if bound_var != var:
                # Don't substitute under bindings of the same variable
                return ForallType(
                    bound_var,
                    self.substitute_type(var, replacement, body)
                )
            case AppType(t1, t2):
                return AppType(
                    self.substitute_type(var, replacement, t1),
                    self.substitute_type(var, replacement, t2)
                )
            case _:
                # For other types, return unchanged
                return ty
    
    def instantiate_left(self, var: str, ty: CoreType) -> None:
        """Instantiate left existential variable: ^alpha <: A."""
        if isinstance(ty, ETVarType) and self.worklist.before(var, ty.name):
            # ^alpha <: ^beta where alpha appears before beta
            self.worklist.solve_evar(ty.name, ETVarType(var))
            return
        elif isinstance(ty, ArrowType):
            # ^alpha <: A -> B  becomes  ^alpha = ^alpha1 -> ^alpha2, A <: ^alpha1, ^alpha2 <: B
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            arrow_ty = ArrowType(ETVarType(a1), ETVarType(a2))
            
            self.worklist.solve_evar(var, arrow_ty)
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            # Note: contravariant in first argument, covariant in second
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(a1))))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a2), ty.t2)))
            return
        elif isinstance(ty, AppType):
            # ^alpha <: F A  becomes  ^alpha = ^alpha1 ^alpha2, ^alpha1 <: F, ^alpha2 <: A
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            app_ty = AppType(ETVarType(a1), ETVarType(a2))
            
            self.worklist.solve_evar(var, app_ty)
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a1), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a2), ty.t2)))
            return
        elif isinstance(ty, ForallType):
            # ^alpha <: forall beta.B  becomes  ►beta,beta |- ^alpha <: B
            fresh_var = self.worklist.fresh_var()
            self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
            substituted = self.substitute_type(ty.var, VarType(fresh_var), ty.ty)
            return self.instantiate_left(var, substituted)
        elif self.is_monotype(ty):
            # For monotypes, just solve directly
            return self.worklist.solve_evar(var, ty)
        else:
            raise InstantiationError(var, ty)
    
    def instantiate_right(self, ty: CoreType, var: str) -> None:
        """Instantiate right existential variable: A <: ^alpha."""
        if isinstance(ty, ETVarType) and self.worklist.before(var, ty.name):
            # ^beta <: ^alpha where alpha appears before beta
            result = self.worklist.solve_evar(ty.name, ETVarType(var))
            return result
        elif isinstance(ty, ArrowType):
            # A -> B <: ^alpha  becomes  ^alpha = ^alpha1 -> ^alpha2, ^alpha1 <: A, B <: ^alpha2
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            arrow_ty = ArrowType(ETVarType(a1), ETVarType(a2))
            
            self.worklist.solve_evar(var, arrow_ty)
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(a1), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(a2))))
            return
        elif isinstance(ty, AppType):
            # F A <: ^alpha  becomes  ^alpha = ^alpha1 ^alpha2, F <: ^alpha1, A <: ^alpha2
            a1 = self.worklist.fresh_evar()
            a2 = self.worklist.fresh_evar()
            app_ty = AppType(ETVarType(a1), ETVarType(a2))
            
            self.worklist.solve_evar(var, app_ty)
            
            self.worklist.push(TyVarEntry(a1, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(a2, ExistentialTyVar()))
            
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(a1))))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(a2))))
            return
        elif isinstance(ty, ForallType):
            # forall beta.B <: ^alpha  becomes  ►^beta,^beta |- [^beta/beta]B <: ^alpha
            fresh_evar = self.worklist.fresh_evar()
            self.worklist.push(TyVarEntry(fresh_evar, MarkerTyVar()))
            self.worklist.push(TyVarEntry(fresh_evar, ExistentialTyVar()))
            substituted = self.substitute_type(ty.var, ETVarType(fresh_evar), ty.ty)
            return self.instantiate_right(substituted, var)
        elif self.is_monotype(ty):
            # For monotypes, just solve directly
            return self.worklist.solve_evar(var, ty)
        else:
            raise InstantiationError(var, ty)
    
    def is_monotype(self, ty: CoreType) -> bool:
        """Check if a type is a monotype (no forall quantifiers)."""
        match ty:
            case ConType(_) | VarType(_) | ETVarType(_):
                return True
            case ArrowType(t1, t2) | AppType(t1, t2):
                return self.is_monotype(t1) and self.is_monotype(t2)
            case ForallType(_, _):
                return False
            case _:
                # For other types like ProductType, assume they're monotypes for now
                return True
    
    def infer_binop_types(self, op: CoreBinOp) -> Tuple[CoreType, CoreType, CoreType]:
        """Infer types for binary operations: (left_type, right_type, result_type)."""
        match op:
            case AddOp() | SubOp() | MulOp() | DivOp():
                # Arithmetic operations: Int -> Int -> Int
                return (ConType("Int"), ConType("Int"), ConType("Int"))
            case LtOp() | LeOp():
                # Comparison operations: Int -> Int -> Bool
                return (ConType("Int"), ConType("Int"), ConType("Bool"))
            case _:
                # Default fallback - shouldn't happen with proper typing
                return (ConType("Int"), ConType("Int"), ConType("Int"))