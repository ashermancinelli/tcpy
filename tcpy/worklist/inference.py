"""DK type inference engine."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core import (
    CoreType, CoreTerm, CorePattern, CoreBinOp, 
    ConType, VarType, ETVarType, ArrowType, ForallType, AppType,
    VarTerm, LitIntTerm, LambdaTerm, AppTerm, TypeLambdaTerm, 
    ConstructorTerm, BinOpTerm, IfTerm, CaseTerm, AddOp, SubOp, MulOp, DivOp, LtOp, LeOp
)
from ..errors import TypeError as TCPyTypeError
from ..errors import (
    UnboundVariableError, UnboundDataConstructorError, NotAFunctionError,
    ArityMismatchError, SubtypingError, InstantiationError
)
from .variables import TyVar, ExistentialTyVar, UniversalTyVar, SolvedTyVar, MarkerTyVar
from .judgments import Judgment, SubJudgment, InfJudgment, ChkJudgment, InfAppJudgment
from .entries import Worklist, TyVarEntry, VarEntry, JudgmentEntry


class DKInference:
    """Complete DK Algorithm type inference engine."""
    
    def __init__(self, data_constructors: Optional[Dict[str, CoreType]] = None,
                 var_context: Optional[Dict[str, CoreType]] = None):
        self.worklist = Worklist()
        self.data_constructors = data_constructors or {}
        self.var_context = var_context or {}
        self.trace: List[str] = []
        
    @classmethod
    def with_context(cls, data_constructors: Dict[str, CoreType],
                     var_context: Dict[str, CoreType]) -> 'DKInference':
        return cls(data_constructors, var_context)
        
    def check_type(self, term: CoreTerm, expected_ty: CoreType) -> None:
        """Check that a term has the expected type."""
        self.worklist.push(JudgmentEntry(ChkJudgment(term, expected_ty)))
        self.solve()
        
    def solve(self) -> None:
        """Main worklist solver loop."""
        while True:
            entry = self.worklist.pop()
            if entry is None:
                break
                
            if isinstance(entry, JudgmentEntry):
                self.solve_judgment(entry.judgment)
            # Other entry types handled elsewhere
            
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
                
    def solve_subtype(self, left: CoreType, right: CoreType) -> None:
        """Solve A <: B subtyping constraint."""
        match (left, right):
            case (VarType(name1), VarType(name2)) if name1 == name2:
                # Reflexivity: a <: a
                return
                
            case (ETVarType(name1), ETVarType(name2)) if name1 == name2:
                # Reflexivity: ^alpha <: ^alpha
                return
                
            case (ConType(name1), ConType(name2)) if name1 == name2:
                # Reflexivity: Int <: Int
                return
                
            case (ETVarType(name), ty):
                # Left instantiation: ^alpha <: A
                self.instantiate_left(name, ty)
                
            case (ty, ETVarType(name)):
                # Right instantiation: A <: ^alpha
                self.instantiate_right(ty, name)
                
            case (ArrowType(a1, b1), ArrowType(a2, b2)):
                # Function subtyping (contravariant in argument)
                self.worklist.push(JudgmentEntry(SubJudgment(a2, a1)))
                self.worklist.push(JudgmentEntry(SubJudgment(b1, b2)))
                
            case (ForallType(var, body), ty):
                # Universal instantiation (left)
                fresh_var = self.worklist.fresh_evar()
                body_instantiated = self.substitute_type(var, ETVarType(fresh_var), body)
                self.worklist.push(TyVarEntry(fresh_var, ExistentialTyVar()))
                self.worklist.push(JudgmentEntry(SubJudgment(body_instantiated, ty)))
                
            case (ty, ForallType(var, body)):
                # Universal instantiation (right)
                fresh_var = self.worklist.fresh_var()
                body_instantiated = self.substitute_type(var, VarType(fresh_var), body)
                self.worklist.push(TyVarEntry(fresh_var, UniversalTyVar()))
                self.worklist.push(JudgmentEntry(SubJudgment(ty, body_instantiated)))
                
            case (AppType(f1, a1), AppType(f2, a2)):
                # Type application subtyping
                self.worklist.push(JudgmentEntry(SubJudgment(f1, f2)))
                self.worklist.push(JudgmentEntry(SubJudgment(a1, a2)))
                
            case _:
                raise SubtypingError(left, right)
                
    def solve_inference(self, term: CoreTerm, ty: CoreType) -> None:
        """Solve type inference judgment: e |- A."""
        self.trace.append(f"Inferring type of {self.term_to_string(term)}")
        
        match term:
            case VarTerm(name):
                # Variable lookup
                if name in self.var_context:
                    var_ty = self.var_context[name]
                else:
                    var_ty = self.worklist.find_var(name)
                    if var_ty is None:
                        raise UnboundVariableError(name)
                
                self.worklist.push(JudgmentEntry(SubJudgment(var_ty, ty)))
                
            case LitIntTerm(value):
                # Integer literal
                self.worklist.push(JudgmentEntry(SubJudgment(ConType("Int"), ty)))
                
            case AppTerm(func, arg):
                # Function application
                fresh_arg_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(fresh_arg_ty.name, ExistentialTyVar()))
                
                # Infer function type as fresh_arg_ty -> ty
                func_ty = ArrowType(fresh_arg_ty, ty)
                self.worklist.push(JudgmentEntry(InfJudgment(func, func_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(arg, fresh_arg_ty)))
                
            case LambdaTerm(param, param_ty, body):
                # Lambda abstraction
                match ty:
                    case ArrowType(arg_ty, ret_ty):
                        # Check parameter type matches
                        self.worklist.push(JudgmentEntry(SubJudgment(arg_ty, param_ty)))
                        
                        # Check body with parameter in scope
                        self.worklist.push(VarEntry(param, param_ty))
                        self.worklist.push(JudgmentEntry(ChkJudgment(body, ret_ty)))
                        
                    case ETVarType(var_name):
                        # Generate fresh arrow type
                        fresh_ret = ETVarType(self.worklist.fresh_evar())
                        arrow_ty = ArrowType(param_ty, fresh_ret)
                        
                        self.worklist.push(TyVarEntry(fresh_ret.name, ExistentialTyVar()))
                        self.instantiate_right(arrow_ty, var_name)
                        
                        # Check body
                        self.worklist.push(VarEntry(param, param_ty))
                        self.worklist.push(JudgmentEntry(ChkJudgment(body, fresh_ret)))
                        
                    case _:
                        raise TypeError(f"Expected function type, got: {ty}")
                        
            case TypeLambdaTerm(param, body):
                # Type abstraction
                match ty:
                    case ForallType(var, body_ty):
                        # Substitute type parameter
                        body_ty_subst = self.substitute_type(var, VarType(param), body_ty)
                        self.worklist.push(JudgmentEntry(ChkJudgment(body, body_ty_subst)))
                        
                    case ETVarType(var_name):
                        # Generate fresh forall type
                        fresh_body = ETVarType(self.worklist.fresh_evar())
                        forall_ty = ForallType(param, fresh_body)
                        
                        self.worklist.push(TyVarEntry(fresh_body.name, ExistentialTyVar()))
                        self.instantiate_right(forall_ty, var_name)
                        self.worklist.push(JudgmentEntry(ChkJudgment(body, fresh_body)))
                        
                    case _:
                        raise TypeError(f"Expected forall type, got: {ty}")
                        
            case ConstructorTerm(name, args):
                # Data constructor
                if name not in self.data_constructors:
                    raise UnboundDataConstructorError(f"Unbound constructor: {name}")
                    
                ctor_ty = self.data_constructors[name]
                
                # Handle constructor application
                if args:
                    # Constructor with arguments - build function type
                    result_ty = ty
                    for arg in reversed(args):
                        fresh_arg_ty = ETVarType(self.worklist.fresh_evar())
                        self.worklist.push(TyVarEntry(fresh_arg_ty.name, ExistentialTyVar()))
                        result_ty = ArrowType(fresh_arg_ty, result_ty)
                        self.worklist.push(JudgmentEntry(ChkJudgment(arg, fresh_arg_ty)))
                    
                    self.worklist.push(JudgmentEntry(SubJudgment(ctor_ty, result_ty)))
                else:
                    # Constructor without arguments
                    self.worklist.push(JudgmentEntry(SubJudgment(ctor_ty, ty)))
                    
            case BinOpTerm(op, left, right):
                # Binary operation
                left_ty, right_ty, result_ty = self.infer_binop_types(op)
                
                self.worklist.push(JudgmentEntry(SubJudgment(result_ty, ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(left, left_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(right, right_ty)))
                
            case IfTerm(cond, then_branch, else_branch):
                # Conditional expression
                self.worklist.push(JudgmentEntry(ChkJudgment(cond, ConType("Bool"))))
                self.worklist.push(JudgmentEntry(ChkJudgment(then_branch, ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(else_branch, ty)))
                
            case CaseTerm(scrutinee, arms):
                # Pattern matching (simplified)
                fresh_scrutinee_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(fresh_scrutinee_ty.name, ExistentialTyVar()))
                self.worklist.push(JudgmentEntry(ChkJudgment(scrutinee, fresh_scrutinee_ty)))
                
                # Check each arm
                for arm in arms:
                    self.worklist.push(JudgmentEntry(ChkJudgment(arm.body, ty)))
                    
            case _:
                raise TypeError(f"Cannot infer type for term: {term}")
                
    def solve_checking(self, term: CoreTerm, ty: CoreType) -> None:
        """Solve type checking judgment: e <= A."""
        self.trace.append(f"Checking {self.term_to_string(term)} against {ty}")
        
        match (term, ty):
            case (LambdaTerm(param, param_ty, body), ArrowType(arg_ty, ret_ty)):
                # Lambda checking against arrow type
                self.worklist.push(JudgmentEntry(SubJudgment(arg_ty, param_ty)))
                self.worklist.push(VarEntry(param, param_ty))
                self.worklist.push(JudgmentEntry(ChkJudgment(body, ret_ty)))
                
            case (TypeLambdaTerm(param, body), ForallType(var, body_ty)):
                # Type lambda checking against forall
                body_ty_subst = self.substitute_type(var, VarType(param), body_ty)
                self.worklist.push(JudgmentEntry(ChkJudgment(body, body_ty_subst)))
                
            case (_, ForallType(var, body)):
                # Checking against forall - instantiate
                fresh_var = self.worklist.fresh_evar()
                body_instantiated = self.substitute_type(var, ETVarType(fresh_var), body)
                self.worklist.push(TyVarEntry(fresh_var, ExistentialTyVar()))
                self.worklist.push(JudgmentEntry(ChkJudgment(term, body_instantiated)))
                
            case _:
                # Default: switch to inference mode
                fresh_ty = ETVarType(self.worklist.fresh_evar())
                self.worklist.push(TyVarEntry(fresh_ty.name, ExistentialTyVar()))
                self.worklist.push(JudgmentEntry(SubJudgment(fresh_ty, ty)))
                self.worklist.push(JudgmentEntry(InfJudgment(term, fresh_ty)))
                
    def solve_inf_app(self, func_ty: CoreType, arg: CoreTerm, result_ty: CoreType) -> None:
        """Solve application synthesis judgment: A * e |- C."""
        self.trace.append(f"InfApp: {func_ty} * {self.term_to_string(arg)} |- {result_ty}")
        match func_ty:
            case ArrowType(param_ty, ret_ty):
                # Function type application
                self.worklist.push(JudgmentEntry(SubJudgment(ret_ty, result_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(arg, param_ty)))
                
            case ETVarType(var_name):
                # Existential application - generate fresh arrow
                fresh_param = ETVarType(self.worklist.fresh_evar())
                fresh_ret = ETVarType(self.worklist.fresh_evar())
                arrow_ty = ArrowType(fresh_param, fresh_ret)
                
                self.worklist.push(TyVarEntry(fresh_param.name, ExistentialTyVar()))
                self.worklist.push(TyVarEntry(fresh_ret.name, ExistentialTyVar()))
                self.instantiate_right(arrow_ty, var_name)
                
                self.worklist.push(JudgmentEntry(SubJudgment(fresh_ret, result_ty)))
                self.worklist.push(JudgmentEntry(ChkJudgment(arg, fresh_param)))
                
            case ForallType(var, body):
                # Polymorphic function application
                fresh_var = self.worklist.fresh_evar()
                body_instantiated = self.substitute_type(var, ETVarType(fresh_var), body)
                self.worklist.push(TyVarEntry(fresh_var, ExistentialTyVar()))
                self.worklist.push(JudgmentEntry(InfAppJudgment(body_instantiated, arg, result_ty)))
                
            case _:
                raise NotAFunctionError(f"Cannot apply non-function type: {func_ty}")
                
    def term_to_string(self, term: CoreTerm) -> str:
        """Convert term to string for tracing."""
        return str(term)
        
    def get_trace(self) -> List[str]:
        """Get inference trace."""
        return self.trace.copy()
        
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
        elif isinstance(ty, ETVarType) and self.worklist.before(ty.name, var):
            # ^beta <: ^alpha where beta appears before alpha
            self.worklist.solve_evar(var, ty)
        elif isinstance(ty, ArrowType):
            # ^alpha <: A -> B
            # Generate ^alpha1 -> ^alpha2 and solve
            fresh_left = self.worklist.fresh_evar()
            fresh_right = self.worklist.fresh_evar()
            
            self.worklist.push(TyVarEntry(fresh_right, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(fresh_left, ExistentialTyVar()))
            
            arrow_ty = ArrowType(ETVarType(fresh_left), ETVarType(fresh_right))
            self.worklist.solve_evar(var, arrow_ty)
            
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(fresh_left))))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(fresh_right), ty.t2)))
            
        elif isinstance(ty, ForallType):
            # ^alpha <: forall a. A
            # Instantiate with fresh existential
            fresh_var = self.worklist.fresh_evar()
            body_instantiated = self.substitute_type(ty.var, ETVarType(fresh_var), ty.ty)
            self.worklist.push(TyVarEntry(fresh_var, ExistentialTyVar()))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(var), body_instantiated)))
            
        elif isinstance(ty, AppType):
            # ^alpha <: F A
            # Generate ^alpha1 ^alpha2 and solve
            fresh_f = self.worklist.fresh_evar()
            fresh_a = self.worklist.fresh_evar()
            
            self.worklist.push(TyVarEntry(fresh_a, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(fresh_f, ExistentialTyVar()))
            
            app_ty = AppType(ETVarType(fresh_f), ETVarType(fresh_a))
            self.worklist.solve_evar(var, app_ty)
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(fresh_f), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(fresh_a), ty.t2)))
            
        elif self.is_monotype(ty):
            # ^alpha <: tau (monotype)
            if not self.occurs_check(var, ty):
                self.worklist.solve_evar(var, ty)
            else:
                raise InstantiationError(f"Occurs check failed: {var} in {ty}")
        else:
            raise InstantiationError(f"Cannot instantiate {var} <: {ty}")
    
    def instantiate_right(self, ty: CoreType, var: str) -> None:
        """Instantiate right existential variable: A <: ^alpha."""
        if isinstance(ty, ETVarType) and self.worklist.before(ty.name, var):
            # ^beta <: ^alpha where beta appears before alpha  
            self.worklist.solve_evar(var, ty)
        elif isinstance(ty, ETVarType) and self.worklist.before(var, ty.name):
            # ^alpha <: ^beta where alpha appears before beta
            self.worklist.solve_evar(ty.name, ETVarType(var))
        elif isinstance(ty, ArrowType):
            # A -> B <: ^alpha
            # Generate ^alpha1 -> ^alpha2 and solve
            fresh_left = self.worklist.fresh_evar()
            fresh_right = self.worklist.fresh_evar()
            
            self.worklist.push(TyVarEntry(fresh_right, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(fresh_left, ExistentialTyVar()))
            
            arrow_ty = ArrowType(ETVarType(fresh_left), ETVarType(fresh_right))
            self.worklist.solve_evar(var, arrow_ty)
            
            self.worklist.push(JudgmentEntry(SubJudgment(ETVarType(fresh_left), ty.t1)))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(fresh_right))))
            
        elif isinstance(ty, ForallType):
            # forall a. A <: ^alpha
            # Add marker and substitute with fresh existential
            marker_var = self.worklist.fresh_var()
            fresh_var = self.worklist.fresh_evar()
            body_instantiated = self.substitute_type(ty.var, ETVarType(fresh_var), ty.ty)
            
            self.worklist.push(TyVarEntry(marker_var, MarkerTyVar()))
            self.worklist.push(TyVarEntry(fresh_var, ExistentialTyVar()))
            self.worklist.push(JudgmentEntry(SubJudgment(body_instantiated, ETVarType(var))))
            
        elif isinstance(ty, AppType):
            # F A <: ^alpha
            # Generate ^alpha1 ^alpha2 and solve
            fresh_f = self.worklist.fresh_evar()
            fresh_a = self.worklist.fresh_evar()
            
            self.worklist.push(TyVarEntry(fresh_a, ExistentialTyVar()))
            self.worklist.push(TyVarEntry(fresh_f, ExistentialTyVar()))
            
            app_ty = AppType(ETVarType(fresh_f), ETVarType(fresh_a))
            self.worklist.solve_evar(var, app_ty)
            
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t1, ETVarType(fresh_f))))
            self.worklist.push(JudgmentEntry(SubJudgment(ty.t2, ETVarType(fresh_a))))
            
        elif self.is_monotype(ty):
            # tau <: ^alpha (monotype)
            if not self.occurs_check(var, ty):
                self.worklist.solve_evar(var, ty)
            else:
                raise InstantiationError(f"Occurs check failed: {var} in {ty}")
        else:
            raise InstantiationError(f"Cannot instantiate {ty} <: {var}")
    
    def is_monotype(self, ty: CoreType) -> bool:
        """Check if a type is a monotype (no forall quantifiers)."""
        match ty:
            case ForallType(_, _):
                return False
            case ArrowType(t1, t2):
                return self.is_monotype(t1) and self.is_monotype(t2)
            case AppType(t1, t2):
                return self.is_monotype(t1) and self.is_monotype(t2)
            case _:
                return True
    
    def infer_binop_types(self, op: CoreBinOp) -> Tuple[CoreType, CoreType, CoreType]:
        """Infer types for binary operations."""
        match op:
            case AddOp() | SubOp() | MulOp() | DivOp():
                # Arithmetic: Int -> Int -> Int
                return (ConType("Int"), ConType("Int"), ConType("Int"))
            case LtOp() | LeOp():
                # Comparison: Int -> Int -> Bool
                return (ConType("Int"), ConType("Int"), ConType("Bool"))
            case _:
                raise TypeError(f"Unknown binary operator: {op}")