"""Tests for the core System F-omega implementation."""

import pytest
from tcpy.core import (
    # Kinds
    StarKind, ArrowKind,
    # Types
    VarType, ConType, ArrowType, ForallType, AppType, ProductType,
    # Terms
    VarTerm, LitIntTerm, LambdaTerm, AppTerm, TypeLambdaTerm,
    ConstructorTerm, BinOpTerm, IfTerm,
    # Binary operations
    AddOp, SubOp, MulOp, DivOp, LtOp, LeOp,
    # Patterns
    WildcardPattern, VarPattern, ConstructorPattern,
    # Module structure
    CoreModule, TypeDef, DataConstructor, TermDef, CaseArm
)


class TestKinds:
    """Test kind system."""
    
    def test_star_kind(self):
        k = StarKind()
        assert str(k) == "*"
    
    def test_arrow_kind(self):
        k = ArrowKind(StarKind(), StarKind())
        assert str(k) == "* -> *"
    
    def test_nested_arrow_kind(self):
        k = ArrowKind(ArrowKind(StarKind(), StarKind()), StarKind())
        assert str(k) == "* -> * -> *"


class TestTypes:
    """Test type system."""
    
    def test_var_type(self):
        t = VarType("a")
        assert str(t) == "a"
    
    def test_con_type(self):
        t = ConType("Int")
        assert str(t) == "Int"
    
    def test_arrow_type(self):
        t = ArrowType(ConType("Int"), ConType("Bool"))
        assert str(t) == "Int -> Bool"
    
    def test_forall_type(self):
        t = ForallType("a", VarType("a"))
        assert str(t) == "∀a. a"
    
    def test_app_type(self):
        t = AppType(ConType("List"), ConType("Int"))
        assert str(t) == "(List Int)"
    
    def test_product_type(self):
        t = ProductType([ConType("Int"), ConType("Bool")])
        assert str(t) == "(Int × Bool)"
    
    def test_empty_product(self):
        t = ProductType([])
        assert str(t) == "()"


class TestTerms:
    """Test term system."""
    
    def test_var_term(self):
        t = VarTerm("x")
        assert str(t) == "x"
    
    def test_lit_int_term(self):
        t = LitIntTerm(42)
        assert str(t) == "42"
    
    def test_lambda_term(self):
        t = LambdaTerm("x", ConType("Int"), VarTerm("x"))
        assert str(t) == "λx : Int. x"
    
    def test_app_term(self):
        t = AppTerm(VarTerm("f"), VarTerm("x"))
        assert str(t) == "f x"
    
    def test_type_lambda_term(self):
        t = TypeLambdaTerm("a", VarTerm("x"))
        assert str(t) == "Λa. x"
    
    def test_constructor_term_no_args(self):
        t = ConstructorTerm("Nil", [])
        assert str(t) == "Nil"
    
    def test_constructor_term_with_args(self):
        t = ConstructorTerm("Cons", [LitIntTerm(1), VarTerm("xs")])
        assert str(t) == "Cons 1 xs"
    
    def test_if_term(self):
        t = IfTerm(VarTerm("b"), LitIntTerm(1), LitIntTerm(0))
        assert str(t) == "if b then 1 else 0"


class TestBinaryOperations:
    """Test binary operations."""
    
    def test_add_op(self):
        op = AddOp()
        assert str(op) == "+"
    
    def test_binop_term(self):
        t = BinOpTerm(AddOp(), LitIntTerm(1), LitIntTerm(2))
        assert str(t) == "1 + 2"
    
    def test_all_binops(self):
        ops = [AddOp(), SubOp(), MulOp(), DivOp(), LtOp(), LeOp()]
        expected = ["+", "-", "*", "/", "<", "<="]
        for op, exp in zip(ops, expected):
            assert str(op) == exp


class TestPatterns:
    """Test pattern system."""
    
    def test_wildcard_pattern(self):
        p = WildcardPattern()
        assert str(p) == "_"
    
    def test_var_pattern(self):
        p = VarPattern("x")
        assert str(p) == "x"
    
    def test_constructor_pattern_no_args(self):
        p = ConstructorPattern("Nil", [])
        assert str(p) == "Nil"
    
    def test_constructor_pattern_with_args(self):
        p = ConstructorPattern("Cons", [VarPattern("x"), VarPattern("xs")])
        assert str(p) == "Cons x xs"


class TestModuleStructure:
    """Test module and definition structures."""
    
    def test_data_constructor(self):
        dc = DataConstructor("Nil", ProductType([]))
        assert dc.name == "Nil"
        assert str(dc.ty) == "()"
    
    def test_type_def(self):
        td = TypeDef("List", ArrowKind(StarKind(), StarKind()), [])
        assert td.name == "List"
        assert str(td.kind) == "* -> *"
    
    def test_term_def(self):
        td = TermDef("id", 
                     ForallType("a", ArrowType(VarType("a"), VarType("a"))),
                     TypeLambdaTerm("a", LambdaTerm("x", VarType("a"), VarTerm("x"))))
        assert td.name == "id"
        assert str(td.ty) == "∀a. a -> a"
    
    def test_core_module(self):
        mod = CoreModule([], [])
        assert len(mod.type_defs) == 0
        assert len(mod.term_defs) == 0


class TestComplexExamples:
    """Test more complex type and term constructions."""
    
    def test_polymorphic_identity(self):
        # ∀a. a -> a
        id_type = ForallType("a", ArrowType(VarType("a"), VarType("a")))
        # Λa. λx:a. x
        id_term = TypeLambdaTerm("a", LambdaTerm("x", VarType("a"), VarTerm("x")))
        
        assert str(id_type) == "∀a. a -> a"
        assert str(id_term) == "Λa. λx : a. x"
    
    def test_list_type_constructor(self):
        # List[Int] represented as (List Int)
        list_int = AppType(ConType("List"), ConType("Int"))
        assert str(list_int) == "(List Int)"
    
    def test_nested_function_type(self):
        # (Int -> Bool) -> List[Int] -> List[Bool]
        t = ArrowType(
            ArrowType(ConType("Int"), ConType("Bool")),
            ArrowType(
                AppType(ConType("List"), ConType("Int")),
                AppType(ConType("List"), ConType("Bool"))
            )
        )
        assert str(t) == "Int -> Bool -> (List Int) -> (List Bool)"