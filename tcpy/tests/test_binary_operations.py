"""Tests for binary operations and their type inference."""

import pytest
from tcpy.core import (
    LitIntTerm, VarTerm, BinOpTerm,
    AddOp, SubOp, MulOp, DivOp, LtOp, LeOp,
    ConType
)
from tcpy.worklist import DKInference
from tcpy.errors import UnboundVariableError


class TestArithmeticOperations:
    """Test arithmetic binary operations."""
    
    def test_addition_integer_literals(self):
        """Test addition of integer literals."""
        dk = DKInference()
        term = BinOpTerm(AddOp(), LitIntTerm(1), LitIntTerm(2))
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_subtraction_integer_literals(self):
        """Test subtraction of integer literals.""" 
        dk = DKInference()
        term = BinOpTerm(SubOp(), LitIntTerm(5), LitIntTerm(3))
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_multiplication_integer_literals(self):
        """Test multiplication of integer literals."""
        dk = DKInference()
        term = BinOpTerm(MulOp(), LitIntTerm(4), LitIntTerm(6))
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_division_integer_literals(self):
        """Test division of integer literals."""
        dk = DKInference()
        term = BinOpTerm(DivOp(), LitIntTerm(10), LitIntTerm(2))
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_arithmetic_with_variables(self):
        """Test arithmetic operations with variables."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int"), "y": ConType("Int")}
        
        # Test each arithmetic operation with variables
        operations = [AddOp(), SubOp(), MulOp(), DivOp()]
        for op in operations:
            term = BinOpTerm(op, VarTerm("x"), VarTerm("y"))
            expected_ty = ConType("Int")
            
            # Create fresh inference context for each test
            dk_fresh = DKInference()
            dk_fresh.var_context = {"x": ConType("Int"), "y": ConType("Int")}
            dk_fresh.check_type(term, expected_ty)
    
    def test_mixed_literal_variable_arithmetic(self):
        """Test arithmetic operations mixing literals and variables."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int")}
        
        # x + 5
        term = BinOpTerm(AddOp(), VarTerm("x"), LitIntTerm(5))
        expected_ty = ConType("Int")
        dk.check_type(term, expected_ty)


class TestComparisonOperations:
    """Test comparison binary operations."""
    
    def test_less_than_integer_literals(self):
        """Test less than comparison of integer literals."""
        dk = DKInference()
        term = BinOpTerm(LtOp(), LitIntTerm(3), LitIntTerm(7))
        expected_ty = ConType("Bool")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_less_equal_integer_literals(self):
        """Test less than or equal comparison of integer literals."""
        dk = DKInference()
        term = BinOpTerm(LeOp(), LitIntTerm(4), LitIntTerm(4))
        expected_ty = ConType("Bool")
        
        dk.check_type(term, expected_ty)
        # Should not raise exception
    
    def test_comparison_with_variables(self):
        """Test comparison operations with variables."""
        dk = DKInference()
        dk.var_context = {"a": ConType("Int"), "b": ConType("Int")}
        
        # Test each comparison operation with variables
        operations = [LtOp(), LeOp()]
        for op in operations:
            term = BinOpTerm(op, VarTerm("a"), VarTerm("b"))
            expected_ty = ConType("Bool")
            
            # Create fresh inference context for each test
            dk_fresh = DKInference()
            dk_fresh.var_context = {"a": ConType("Int"), "b": ConType("Int")}
            dk_fresh.check_type(term, expected_ty)
    
    def test_mixed_literal_variable_comparison(self):
        """Test comparison operations mixing literals and variables."""
        dk = DKInference()
        dk.var_context = {"num": ConType("Int")}
        
        # num < 10
        term = BinOpTerm(LtOp(), VarTerm("num"), LitIntTerm(10))
        expected_ty = ConType("Bool")
        dk.check_type(term, expected_ty)


class TestNestedBinaryOperations:
    """Test nested and complex binary operations."""
    
    def test_nested_arithmetic(self):
        """Test nested arithmetic operations."""
        dk = DKInference()
        
        # (1 + 2) * 3
        inner_add = BinOpTerm(AddOp(), LitIntTerm(1), LitIntTerm(2))
        term = BinOpTerm(MulOp(), inner_add, LitIntTerm(3))
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
    
    def test_nested_comparisons_and_arithmetic(self):
        """Test mixing arithmetic and comparison operations."""
        dk = DKInference()
        
        # (5 + 3) < (4 * 2)
        left_add = BinOpTerm(AddOp(), LitIntTerm(5), LitIntTerm(3))
        right_mul = BinOpTerm(MulOp(), LitIntTerm(4), LitIntTerm(2))
        term = BinOpTerm(LtOp(), left_add, right_mul)
        expected_ty = ConType("Bool")
        
        dk.check_type(term, expected_ty)
    
    def test_deeply_nested_operations(self):
        """Test deeply nested binary operations."""
        dk = DKInference()
        
        # ((1 + 2) - 3) * ((4 / 2) + 1)
        left_inner = BinOpTerm(AddOp(), LitIntTerm(1), LitIntTerm(2))
        left_outer = BinOpTerm(SubOp(), left_inner, LitIntTerm(3))
        
        right_inner = BinOpTerm(DivOp(), LitIntTerm(4), LitIntTerm(2))
        right_outer = BinOpTerm(AddOp(), right_inner, LitIntTerm(1))
        
        term = BinOpTerm(MulOp(), left_outer, right_outer)
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
    
    def test_complex_comparison_chain(self):
        """Test complex comparison with arithmetic subexpressions."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int"), "y": ConType("Int")}
        
        # (x + 5) <= (y * 2)
        left_expr = BinOpTerm(AddOp(), VarTerm("x"), LitIntTerm(5))
        right_expr = BinOpTerm(MulOp(), VarTerm("y"), LitIntTerm(2))
        term = BinOpTerm(LeOp(), left_expr, right_expr)
        expected_ty = ConType("Bool")
        
        dk.check_type(term, expected_ty)


class TestBinaryOperationErrorCases:
    """Test error conditions with binary operations."""
    
    def test_unbound_variable_in_binary_op(self):
        """Test error when using unbound variable in binary operation."""
        dk = DKInference()
        
        # unknown_var + 5
        term = BinOpTerm(AddOp(), VarTerm("unknown_var"), LitIntTerm(5))
        expected_ty = ConType("Int")
        
        with pytest.raises(UnboundVariableError):
            dk.check_type(term, expected_ty)
    
    def test_partially_unbound_binary_op(self):
        """Test error when one operand is bound and other is unbound."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int")}
        
        # x + unknown_var
        term = BinOpTerm(AddOp(), VarTerm("x"), VarTerm("unknown_var"))
        expected_ty = ConType("Int")
        
        with pytest.raises(UnboundVariableError):
            dk.check_type(term, expected_ty)


class TestBinaryOperationTypes:
    """Test different aspects of binary operation typing."""
    
    def test_operation_type_consistency(self):
        """Test that operations maintain proper type consistency."""
        dk = DKInference()
        
        # Arithmetic operations should produce numeric results
        arithmetic_ops = [AddOp(), SubOp(), MulOp(), DivOp()]
        for op in arithmetic_ops:
            term = BinOpTerm(op, LitIntTerm(1), LitIntTerm(2))
            expected_ty = ConType("Int")
            
            dk_fresh = DKInference()
            dk_fresh.check_type(term, expected_ty)
            # Should not raise exception
    
    def test_comparison_type_consistency(self):
        """Test that comparisons produce boolean results."""
        dk = DKInference()
        
        # Comparison operations should produce boolean results
        comparison_ops = [LtOp(), LeOp()]
        for op in comparison_ops:
            term = BinOpTerm(op, LitIntTerm(1), LitIntTerm(2))
            expected_ty = ConType("Bool")
            
            dk_fresh = DKInference()
            dk_fresh.check_type(term, expected_ty)
            # Should not raise exception


class TestBinaryOperationStringRepresentation:
    """Test string representation of binary operations."""
    
    def test_arithmetic_op_string_representation(self):
        """Test string representation of arithmetic operations."""
        operations = [
            (AddOp(), "+"),
            (SubOp(), "-"), 
            (MulOp(), "*"),
            (DivOp(), "/")
        ]
        
        for op, expected_str in operations:
            assert str(op) == expected_str
    
    def test_comparison_op_string_representation(self):
        """Test string representation of comparison operations."""
        operations = [
            (LtOp(), "<"),
            (LeOp(), "<=")
        ]
        
        for op, expected_str in operations:
            assert str(op) == expected_str
    
    def test_binary_term_string_representation(self):
        """Test string representation of complete binary terms."""
        # 1 + 2
        term = BinOpTerm(AddOp(), LitIntTerm(1), LitIntTerm(2))
        assert str(term) == "1 + 2"
        
        # x < y
        term2 = BinOpTerm(LtOp(), VarTerm("x"), VarTerm("y"))
        assert str(term2) == "x < y"
        
        # Nested: (a + b) * c
        inner = BinOpTerm(AddOp(), VarTerm("a"), VarTerm("b"))
        outer = BinOpTerm(MulOp(), inner, VarTerm("c"))
        assert str(outer) == "a + b * c"


class TestBinaryOperationIntegration:
    """Integration tests for binary operations with other language constructs."""
    
    def test_binary_ops_with_if_expressions(self):
        """Test binary operations combined with if expressions."""
        from tcpy.core import IfTerm
        
        dk = DKInference()
        dk.var_context = {"x": ConType("Int"), "y": ConType("Int")}
        
        # if (x < y) then (x + 1) else (y - 1)
        condition = BinOpTerm(LtOp(), VarTerm("x"), VarTerm("y"))
        then_branch = BinOpTerm(AddOp(), VarTerm("x"), LitIntTerm(1))
        else_branch = BinOpTerm(SubOp(), VarTerm("y"), LitIntTerm(1))
        
        if_term = IfTerm(condition, then_branch, else_branch)
        expected_ty = ConType("Int")
        
        dk.check_type(if_term, expected_ty)
    
    def test_binary_ops_in_function_application_context(self):
        """Test binary operations as arguments to function application."""
        from tcpy.core import LambdaTerm, AppTerm
        
        dk = DKInference()
        
        # (lambda x:Int. x + 1) (2 * 3)
        lambda_body = BinOpTerm(AddOp(), VarTerm("x"), LitIntTerm(1))
        lambda_term = LambdaTerm("x", ConType("Int"), lambda_body)
        
        arg_term = BinOpTerm(MulOp(), LitIntTerm(2), LitIntTerm(3))
        app_term = AppTerm(lambda_term, arg_term)
        
        expected_ty = ConType("Int")
        dk.check_type(app_term, expected_ty)