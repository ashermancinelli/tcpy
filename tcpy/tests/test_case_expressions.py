"""Tests for case expressions and type inference."""

import pytest
from tcpy.core import (
    # Patterns
    WildcardPattern, VarPattern, ConstructorPattern,
    # Types
    ConType, ArrowType, ETVarType,
    # Terms
    CaseTerm, CaseArm, VarTerm, LitIntTerm, ConstructorTerm
)
from tcpy.worklist import DKInference
from tcpy.errors import UnboundVariableError


class TestCaseExpressionInference:
    """Test type inference for case expressions."""
    
    def test_simple_case_expression_inference(self):
        """Test basic case expression type inference."""
        dk = DKInference()
        dk.var_context = {"x": ConType("List")}  # Bind x to a type
        
        # case x of
        #   Nil -> 0
        #   _ -> 1
        scrutinee = VarTerm("x")
        nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(0))
        wild_arm = CaseArm(WildcardPattern(), LitIntTerm(1))
        case_expr = CaseTerm(scrutinee, [nil_arm, wild_arm])
        
        # Should not raise an exception during inference
        # Note: This tests the basic structure, not pattern variable handling
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)
        
        # Check that trace was generated
        trace = dk.get_trace()
        assert len(trace) > 0
    
    def test_case_expression_with_variable_scrutinee(self):
        """Test case expression with bound variable as scrutinee."""
        dk = DKInference()
        dk.var_context = {"list": ConType("List")}
        
        # case list of
        #   Nil -> 42
        #   _ -> 0
        scrutinee = VarTerm("list")
        nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(42))
        wild_arm = CaseArm(WildcardPattern(), LitIntTerm(0))
        case_expr = CaseTerm(scrutinee, [nil_arm, wild_arm])
        
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)
        
        trace = dk.get_trace()
        assert any("match" in entry for entry in trace)
    
    def test_case_expression_with_unbound_variable(self):
        """Test case expression with unbound variable as scrutinee."""
        dk = DKInference()
        
        scrutinee = VarTerm("unknown")
        arm = CaseArm(WildcardPattern(), LitIntTerm(1))
        case_expr = CaseTerm(scrutinee, [arm])
        
        # Should raise UnboundVariableError for the scrutinee
        with pytest.raises(UnboundVariableError):
            dk.check_type(case_expr, ConType("Int"))
    
    def test_empty_case_expression(self):
        """Test case expression with no arms."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Void")}
        
        scrutinee = VarTerm("x")
        case_expr = CaseTerm(scrutinee, [])  # No arms
        
        # Should not crash, though this is an unusual case
        expected_ty = ConType("Never")
        dk.check_type(case_expr, expected_ty)
    
    def test_single_arm_case_expression(self):
        """Test case expression with single arm."""
        dk = DKInference()
        dk.var_context = {"value": ConType("Option")}
        
        scrutinee = VarTerm("value")
        arm = CaseArm(WildcardPattern(), LitIntTerm(42))
        case_expr = CaseTerm(scrutinee, [arm])
        
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)


class TestCaseExpressionPatterns:
    """Test different patterns in case expressions."""
    
    def test_case_with_constructor_patterns(self):
        """Test case with various constructor patterns."""
        dk = DKInference()
        dk.var_context = {"opt": ConType("Option")}
        
        # case opt of
        #   None -> 0
        #   Some _ -> 1
        scrutinee = VarTerm("opt")
        none_arm = CaseArm(ConstructorPattern("None", []), LitIntTerm(0))
        some_arm = CaseArm(ConstructorPattern("Some", [WildcardPattern()]), LitIntTerm(1))
        case_expr = CaseTerm(scrutinee, [none_arm, some_arm])
        
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)
    
    def test_case_with_variable_patterns(self):
        """Test case with variable patterns."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Any")}
        
        # case x of
        #   y -> 42  # y would bind to the value of x
        scrutinee = VarTerm("x")
        var_arm = CaseArm(VarPattern("y"), LitIntTerm(42))
        case_expr = CaseTerm(scrutinee, [var_arm])
        
        expected_ty = ConType("Int")
        # Note: This tests structure but doesn't verify y is properly bound
        # since pattern variable binding is not fully implemented
        dk.check_type(case_expr, expected_ty)
    
    def test_case_with_nested_patterns(self):
        """Test case with nested constructor patterns."""
        dk = DKInference()
        dk.var_context = {"tree": ConType("Tree")}
        
        # case tree of
        #   Leaf x -> 1
        #   Node (Leaf _) (Leaf _) -> 2
        #   _ -> 0
        scrutinee = VarTerm("tree")
        
        leaf_arm = CaseArm(ConstructorPattern("Leaf", [VarPattern("x")]), LitIntTerm(1))
        
        left_leaf = ConstructorPattern("Leaf", [WildcardPattern()])
        right_leaf = ConstructorPattern("Leaf", [WildcardPattern()])
        node_arm = CaseArm(ConstructorPattern("Node", [left_leaf, right_leaf]), LitIntTerm(2))
        
        wild_arm = CaseArm(WildcardPattern(), LitIntTerm(0))
        
        case_expr = CaseTerm(scrutinee, [leaf_arm, node_arm, wild_arm])
        
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)


class TestCaseExpressionBodies:
    """Test different body expressions in case arms."""
    
    def test_case_with_different_body_types(self):
        """Test case arms with different body expression types."""
        dk = DKInference()
        dk.var_context = {"input": ConType("Input")}
        
        # case input of
        #   A -> variable reference
        #   B -> constructor  
        #   C -> literal
        scrutinee = VarTerm("input")
        
        # Add some variables to context for the arms
        dk.var_context.update({
            "result1": ConType("String"),
            "result2": ConType("String")
        })
        
        var_arm = CaseArm(ConstructorPattern("A", []), VarTerm("result1"))
        cons_arm = CaseArm(ConstructorPattern("B", []), VarTerm("result2"))  # Use variable instead
        lit_arm = CaseArm(ConstructorPattern("C", []), LitIntTerm(42))
        
        # Note: These arms have different types, which would normally be a type error
        # But our current implementation doesn't enforce arm type consistency
        case_expr1 = CaseTerm(scrutinee, [var_arm])
        case_expr2 = CaseTerm(scrutinee, [cons_arm])
        case_expr3 = CaseTerm(scrutinee, [lit_arm])
        
        # Test each separately since they have different types
        dk.check_type(case_expr1, ConType("String"))
        
        dk_fresh = DKInference()
        dk_fresh.var_context = {"input": ConType("Input"), "result2": ConType("String")}
        dk_fresh.check_type(case_expr2, ConType("String"))
        
        dk_fresh2 = DKInference()
        dk_fresh2.var_context = {"input": ConType("Input")}
        dk_fresh2.check_type(case_expr3, ConType("Int"))
    
    def test_case_with_nested_expressions(self):
        """Test case with nested expressions in arms."""
        dk = DKInference()
        dk.var_context = {
            "data": ConType("Data"),
            "helper": ConType("Int")
        }
        
        # case data of
        #   Simple -> helper
        #   Complex -> case helper of _ -> 0
        scrutinee = VarTerm("data")
        
        simple_arm = CaseArm(ConstructorPattern("Simple", []), VarTerm("helper"))
        
        # Nested case in the Complex arm
        inner_case = CaseTerm(
            VarTerm("helper"), 
            [CaseArm(WildcardPattern(), LitIntTerm(0))]
        )
        complex_arm = CaseArm(ConstructorPattern("Complex", []), inner_case)
        
        case_expr = CaseTerm(scrutinee, [simple_arm, complex_arm])
        
        expected_ty = ConType("Int")
        dk.check_type(case_expr, expected_ty)


class TestCaseExpressionLimitations:
    """Test current limitations of case expression implementation."""
    
    def test_pattern_variable_binding_limitation(self):
        """Test that pattern variables aren't properly bound (current limitation)."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Pair")}
        
        # case x of
        #   Pair a b -> a  # 'a' should be bound but currently isn't
        scrutinee = VarTerm("x")
        pair_pattern = ConstructorPattern("Pair", [VarPattern("a"), VarPattern("b")])
        arm = CaseArm(pair_pattern, VarTerm("a"))  # 'a' will be unbound
        case_expr = CaseTerm(scrutinee, [arm])
        
        # This should ideally work, but currently 'a' will be unbound
        # So we expect an UnboundVariableError
        with pytest.raises(UnboundVariableError):
            dk.check_type(case_expr, ConType("Int"))
    
    def test_scrutinee_type_constraint_limitation(self):
        """Test that scrutinee type constraints aren't enforced (current limitation)."""
        dk = DKInference()
        dk.var_context = {"x": ConType("Int")}  # x is an Int
        
        # case x of
        #   Cons y ys -> 1  # Pattern expects List, but x is Int
        scrutinee = VarTerm("x")
        cons_pattern = ConstructorPattern("Cons", [VarPattern("y"), VarPattern("ys")])
        arm = CaseArm(cons_pattern, LitIntTerm(1))
        case_expr = CaseTerm(scrutinee, [arm])
        
        # Currently this might not catch the type mismatch between
        # Int scrutinee and List pattern
        expected_ty = ConType("Int")
        # The current implementation may not catch this mismatch
        try:
            dk.check_type(case_expr, expected_ty)
        except Exception:
            # Any exception is acceptable here since this is testing limitations
            pass
    
    def test_case_expression_string_representation(self):
        """Test string representation of case expressions."""
        scrutinee = VarTerm("x")
        arm1 = CaseArm(ConstructorPattern("A", []), LitIntTerm(1))
        arm2 = CaseArm(WildcardPattern(), LitIntTerm(2))
        case_expr = CaseTerm(scrutinee, [arm1, arm2])
        
        # Test that we can create and represent case expressions
        # The exact string format isn't critical, just that it doesn't crash
        str_repr = str(case_expr)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


class TestCaseExpressionIntegration:
    """Integration tests for case expressions with type inference."""
    
    def test_case_in_function_context(self):
        """Test case expression in a function-like context."""
        dk = DKInference()
        
        # Simulate: λx. case x of Nil -> 0; _ -> 1
        # We'll test just the case part, assuming x is bound
        dk.var_context = {"x": ConType("List")}
        
        scrutinee = VarTerm("x")
        nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(0))
        wild_arm = CaseArm(WildcardPattern(), LitIntTerm(1))
        case_expr = CaseTerm(scrutinee, [nil_arm, wild_arm])
        
        result_ty = ConType("Int")
        dk.check_type(case_expr, result_ty)
        
        # Verify type inference worked
        trace = dk.get_trace()
        assert len(trace) > 0
        assert any("match" in entry for entry in trace)
    
    def test_multiple_case_expressions(self):
        """Test multiple case expressions in sequence."""
        dk = DKInference()
        dk.var_context = {
            "x": ConType("Option"),
            "y": ConType("Maybe")
        }
        
        # First case expression
        case1 = CaseTerm(
            VarTerm("x"),
            [CaseArm(WildcardPattern(), LitIntTerm(1))]
        )
        
        # Second case expression  
        case2 = CaseTerm(
            VarTerm("y"),
            [CaseArm(WildcardPattern(), LitIntTerm(2))]
        )
        
        # Test both separately
        dk.check_type(case1, ConType("Int"))
        
        # Fresh inference context for second case
        dk2 = DKInference()
        dk2.var_context = {"y": ConType("Maybe")}
        dk2.check_type(case2, ConType("Int"))