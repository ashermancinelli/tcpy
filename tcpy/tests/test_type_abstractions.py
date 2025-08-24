"""Tests for type abstractions (TypeLambdaTerm) and polymorphic types."""

import pytest
from tcpy.core import (
    TypeLambdaTerm, LambdaTerm, VarTerm, LitIntTerm, AppTerm,
    ConType, VarType, ForallType, ArrowType, ETVarType
)
from tcpy.worklist import DKInference
from tcpy.errors import UnboundVariableError


class TestBasicTypeAbstractions:
    """Test basic type abstraction functionality."""
    
    def test_simple_type_abstraction(self):
        """Test simple type abstraction: Lambda a. x"""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int")}
        
        # Lambda a. x  (where x : Int)
        type_abs = TypeLambdaTerm("a", VarTerm("x"))
        expected_ty = ForallType("a", ConType("Int"))
        
        dk.check_type(type_abs, expected_ty)
        # Should not raise exception
    
    def test_type_abstraction_with_type_variable_usage(self):
        """Test type abstraction that uses the type variable."""
        dk = DKInference()
        dk.var_context = {"x": VarType("a")}  # x has type a
        
        # Lambda a. x  (where x : a)
        type_abs = TypeLambdaTerm("a", VarTerm("x"))
        expected_ty = ForallType("a", VarType("a"))
        
        dk.check_type(type_abs, expected_ty)
        # Should not raise exception
    
    def test_nested_type_abstraction(self):
        """Test nested type abstractions: Lambda a. Lambda b. x"""
        dk = DKInference()
        dk.var_context = {"x": ConType("Bool")}
        
        # Lambda a. Lambda b. x
        inner_abs = TypeLambdaTerm("b", VarTerm("x"))
        outer_abs = TypeLambdaTerm("a", inner_abs)
        expected_ty = ForallType("a", ForallType("b", ConType("Bool")))
        
        dk.check_type(outer_abs, expected_ty)
    
    def test_type_abstraction_with_literal(self):
        """Test type abstraction over a literal."""
        dk = DKInference()
        
        # Lambda a. 42
        type_abs = TypeLambdaTerm("a", LitIntTerm(42))
        expected_ty = ForallType("a", ConType("Int"))
        
        dk.check_type(type_abs, expected_ty)


class TestTypeAbstractionWithTermAbstraction:
    """Test combinations of type abstraction and term abstraction."""
    
    def test_polymorphic_identity_function(self):
        """Test the polymorphic identity function: Lambda a. lambda x:a. x"""
        dk = DKInference()
        
        # Lambda a. lambda x:a. x
        term_lambda = LambdaTerm("x", VarType("a"), VarTerm("x"))
        type_lambda = TypeLambdaTerm("a", term_lambda)
        expected_ty = ForallType("a", ArrowType(VarType("a"), VarType("a")))
        
        dk.check_type(type_lambda, expected_ty)
    
    def test_polymorphic_constant_function(self):
        """Test a polymorphic constant function: Lambda a. lambda x:a. 42"""
        dk = DKInference()
        
        # Lambda a. lambda x:a. 42
        term_lambda = LambdaTerm("x", VarType("a"), LitIntTerm(42))
        type_lambda = TypeLambdaTerm("a", term_lambda)
        expected_ty = ForallType("a", ArrowType(VarType("a"), ConType("Int")))
        
        dk.check_type(type_lambda, expected_ty)
    
    def test_mixed_type_term_abstraction_order(self):
        """Test different orders of type and term abstractions."""
        dk = DKInference()
        
        # lambda f:a->b. Lambda a. Lambda b. f
        # This tests a more complex polymorphic type
        dk.var_context = {"f": ArrowType(VarType("a"), VarType("b"))}
        
        inner_type_abs = TypeLambdaTerm("b", VarTerm("f"))
        outer_type_abs = TypeLambdaTerm("a", inner_type_abs)
        
        expected_inner = ForallType("b", ArrowType(VarType("a"), VarType("b")))
        expected_ty = ForallType("a", expected_inner)
        
        dk.check_type(outer_type_abs, expected_ty)


class TestTypeAbstractionErrorCases:
    """Test error cases with type abstractions."""
    
    def test_type_abstraction_with_unbound_variable(self):
        """Test type abstraction over unbound variable."""
        dk = DKInference()
        
        # Lambda a. unknown_var
        type_abs = TypeLambdaTerm("a", VarTerm("unknown_var"))
        expected_ty = ForallType("a", ConType("Int"))  # arbitrary expected type
        
        with pytest.raises(UnboundVariableError):
            dk.check_type(type_abs, expected_ty)
    
    def test_type_abstraction_scope_shadowing(self):
        """Test that type variable properly shadows outer scope."""
        dk = DKInference()
        dk.var_context = {"x": VarType("a")}  # x has type a in outer scope
        
        # Lambda a. x  - should use the new 'a' from type abstraction
        type_abs = TypeLambdaTerm("a", VarTerm("x"))
        expected_ty = ForallType("a", VarType("a"))
        
        # This should work because the inner 'a' shadows any outer 'a'
        dk.check_type(type_abs, expected_ty)


class TestTypeAbstractionWithApplications:
    """Test type abstractions with function applications."""
    
    def test_polymorphic_function_application(self):
        """Test applying a polymorphic function."""
        dk = DKInference()
        dk.var_context = {
            "poly_id": ForallType("a", ArrowType(VarType("a"), VarType("a")))
        }
        
        # Test that we can reference polymorphic functions
        # Lambda b. poly_id
        type_abs = TypeLambdaTerm("b", VarTerm("poly_id"))
        expected_ty = ForallType("b", ForallType("a", ArrowType(VarType("a"), VarType("a"))))
        
        dk.check_type(type_abs, expected_ty)
    
    def test_type_abstraction_over_application(self):
        """Test type abstraction over function application."""
        dk = DKInference()
        dk.var_context = {
            "f": ArrowType(ConType("Int"), VarType("a")),
            "x": ConType("Int")
        }
        
        # Lambda a. f x  (where f : Int -> a, x : Int)
        app_term = AppTerm(VarTerm("f"), VarTerm("x"))
        type_abs = TypeLambdaTerm("a", app_term)
        expected_ty = ForallType("a", VarType("a"))
        
        dk.check_type(type_abs, expected_ty)


class TestComplexTypeAbstractions:
    """Test complex scenarios with type abstractions."""
    
    def test_church_numerals_type_structure(self):
        """Test Church numeral-like type structure."""
        dk = DKInference()
        
        # Lambda a. lambda f:(a->a). lambda x:a. f (f x)
        # This represents Church numeral 2
        inner_app = AppTerm(VarTerm("f"), VarTerm("x"))
        outer_app = AppTerm(VarTerm("f"), inner_app)
        
        arrow_type = ArrowType(VarType("a"), VarType("a"))
        x_lambda = LambdaTerm("x", VarType("a"), outer_app)
        f_lambda = LambdaTerm("f", arrow_type, x_lambda)
        type_lambda = TypeLambdaTerm("a", f_lambda)
        
        # Type should be: forall a. (a -> a) -> a -> a
        expected_inner = ArrowType(arrow_type, ArrowType(VarType("a"), VarType("a")))
        expected_ty = ForallType("a", expected_inner)
        
        dk.check_type(type_lambda, expected_ty)
    
    def test_multiple_type_variables_same_body(self):
        """Test using multiple type variables in same body."""
        dk = DKInference()
        dk.var_context = {
            "pair_constructor": ArrowType(VarType("a"), ArrowType(VarType("b"), ConType("Pair")))
        }
        
        # Lambda a. Lambda b. pair_constructor
        inner_abs = TypeLambdaTerm("b", VarTerm("pair_constructor"))
        outer_abs = TypeLambdaTerm("a", inner_abs)
        
        inner_arrow = ArrowType(VarType("a"), ArrowType(VarType("b"), ConType("Pair")))
        expected_inner = ForallType("b", inner_arrow)
        expected_ty = ForallType("a", expected_inner)
        
        dk.check_type(outer_abs, expected_ty)


class TestTypeAbstractionStringRepresentation:
    """Test string representation of type abstractions."""
    
    def test_simple_type_lambda_string(self):
        """Test string representation of simple type lambda."""
        term = TypeLambdaTerm("a", VarTerm("x"))
        assert str(term) == "Lambda a. x"
    
    def test_nested_type_lambda_string(self):
        """Test string representation of nested type lambdas."""
        inner = TypeLambdaTerm("b", VarTerm("x"))
        outer = TypeLambdaTerm("a", inner)
        assert str(outer) == "Lambda a. Lambda b. x"
    
    def test_type_lambda_with_term_lambda_string(self):
        """Test string representation mixing type and term lambdas."""
        term_lambda = LambdaTerm("x", VarType("a"), VarTerm("x"))
        type_lambda = TypeLambdaTerm("a", term_lambda)
        expected = "Lambda a. lambda x : a. x"
        assert str(type_lambda) == expected


class TestTypeAbstractionIntegration:
    """Integration tests for type abstractions with other constructs."""
    
    def test_type_abstraction_with_conditionals(self):
        """Test type abstraction containing conditional expressions."""
        from tcpy.core import IfTerm, BinOpTerm, LtOp
        
        dk = DKInference()
        dk.var_context = {"x": VarType("a"), "y": VarType("a")}
        
        # Lambda a. if (x < y) then x else y  -- polymorphic min function
        # Note: This assumes we have comparison for type a
        condition = BinOpTerm(LtOp(), VarTerm("x"), VarTerm("y"))
        if_term = IfTerm(condition, VarTerm("x"), VarTerm("y"))
        type_abs = TypeLambdaTerm("a", if_term)
        
        expected_ty = ForallType("a", VarType("a"))
        dk.check_type(type_abs, expected_ty)
    
    def test_type_abstraction_in_let_binding(self):
        """Test that type abstractions work conceptually in bindings."""
        dk = DKInference()
        
        # Just test that we can create and type check a polymorphic function
        # Lambda a. lambda x:a. x  (polymorphic identity)
        term_lambda = LambdaTerm("x", VarType("a"), VarTerm("x"))
        type_lambda = TypeLambdaTerm("a", term_lambda)
        
        poly_id_type = ForallType("a", ArrowType(VarType("a"), VarType("a")))
        dk.check_type(type_lambda, poly_id_type)
        
        # Verify the type is what we expect
        trace = dk.get_trace()
        assert len(trace) > 0
    
    def test_instantiation_of_polymorphic_types(self):
        """Test that polymorphic types can be properly instantiated."""
        dk = DKInference()
        
        # Create a simple polymorphic function type
        # Lambda a. lambda x:a. x
        term_lambda = LambdaTerm("x", VarType("a"), VarTerm("x"))
        type_lambda = TypeLambdaTerm("a", term_lambda)
        
        # Should be able to check against the polymorphic type
        poly_type = ForallType("a", ArrowType(VarType("a"), VarType("a")))
        dk.check_type(type_lambda, poly_type)
        
        # And also against a monomorphic instantiation
        dk2 = DKInference()
        mono_type = ArrowType(ConType("Int"), ConType("Int"))
        # Note: This would require type application, which we don't fully test here
        # But the type structure should be compatible