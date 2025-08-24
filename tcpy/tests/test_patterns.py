"""Tests for pattern matching system."""

import pytest
from tcpy.core import (
    # Patterns
    CorePattern, WildcardPattern, VarPattern, ConstructorPattern,
    # Types
    ConType, ArrowType, ProductType,
    # Terms
    CaseTerm, CaseArm, VarTerm, LitIntTerm, ConstructorTerm
)


class TestPatternBasics:
    """Test basic pattern functionality."""
    
    def test_wildcard_pattern(self):
        """Test wildcard pattern creation and string representation."""
        pat = WildcardPattern()
        assert str(pat) == "_"
        assert isinstance(pat, CorePattern)
    
    def test_var_pattern(self):
        """Test variable pattern creation and string representation."""
        pat = VarPattern("x")
        assert pat.name == "x"
        assert str(pat) == "x"
        assert isinstance(pat, CorePattern)
    
    def test_constructor_pattern_no_args(self):
        """Test constructor pattern with no arguments."""
        pat = ConstructorPattern("Nil", [])
        assert pat.name == "Nil"
        assert pat.args == []
        assert str(pat) == "Nil"
        assert isinstance(pat, CorePattern)
    
    def test_constructor_pattern_with_args(self):
        """Test constructor pattern with arguments."""
        args = [VarPattern("x"), WildcardPattern()]
        pat = ConstructorPattern("Cons", args)
        assert pat.name == "Cons"
        assert pat.args == args
        assert str(pat) == "Cons x _"
        assert isinstance(pat, CorePattern)


class TestNestedPatterns:
    """Test nested pattern structures."""
    
    def test_nested_constructor_patterns(self):
        """Test deeply nested constructor patterns."""
        # Cons x (Cons y Nil)
        inner_cons = ConstructorPattern("Cons", [VarPattern("y"), ConstructorPattern("Nil", [])])
        outer_cons = ConstructorPattern("Cons", [VarPattern("x"), inner_cons])
        
        assert str(outer_cons) == "Cons x Cons y Nil"
        assert outer_cons.name == "Cons"
        assert len(outer_cons.args) == 2
        assert isinstance(outer_cons.args[0], VarPattern)
        assert isinstance(outer_cons.args[1], ConstructorPattern)
    
    def test_pattern_with_wildcards_and_vars(self):
        """Test patterns mixing wildcards and variables."""
        # Tuple (_, x, _)
        args = [WildcardPattern(), VarPattern("x"), WildcardPattern()]
        pat = ConstructorPattern("Tuple", args)
        
        assert str(pat) == "Tuple _ x _"
        assert len(pat.args) == 3
        assert isinstance(pat.args[0], WildcardPattern)
        assert isinstance(pat.args[1], VarPattern)
        assert isinstance(pat.args[2], WildcardPattern)
    
    def test_complex_nested_pattern(self):
        """Test complex nested patterns."""
        # Either (Left x) (Right (Cons y _))
        left_pattern = ConstructorPattern("Left", [VarPattern("x")])
        right_inner = ConstructorPattern("Cons", [VarPattern("y"), WildcardPattern()])
        right_pattern = ConstructorPattern("Right", [right_inner])
        either_pattern = ConstructorPattern("Either", [left_pattern, right_pattern])
        
        expected = "Either Left x Right Cons y _"
        assert str(either_pattern) == expected


class TestCaseArms:
    """Test case expression arms."""
    
    def test_simple_case_arm(self):
        """Test simple case arm creation."""
        pat = VarPattern("x")
        body = VarTerm("x")
        arm = CaseArm(pat, body)
        
        assert arm.pattern == pat
        assert arm.body == body
        assert isinstance(arm.pattern, VarPattern)
        assert isinstance(arm.body, VarTerm)
    
    def test_case_arm_with_constructor_pattern(self):
        """Test case arm with constructor pattern."""
        pat = ConstructorPattern("Cons", [VarPattern("head"), VarPattern("tail")])
        body = VarTerm("head")
        arm = CaseArm(pat, body)
        
        assert arm.pattern == pat
        assert arm.body == body
        assert str(arm.pattern) == "Cons head tail"
    
    def test_case_arm_with_wildcard(self):
        """Test case arm with wildcard pattern."""
        pat = WildcardPattern()
        body = LitIntTerm(0)
        arm = CaseArm(pat, body)
        
        assert arm.pattern == pat
        assert arm.body == body
        assert str(arm.pattern) == "_"


class TestCaseExpressions:
    """Test complete case expressions."""
    
    def test_simple_case_expression(self):
        """Test simple case expression."""
        scrutinee = VarTerm("x")
        
        # case x of
        #   Nil -> 0
        #   Cons y _ -> 1
        nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(0))
        cons_arm = CaseArm(
            ConstructorPattern("Cons", [VarPattern("y"), WildcardPattern()]),
            LitIntTerm(1)
        )
        
        case_expr = CaseTerm(scrutinee, [nil_arm, cons_arm])
        
        assert case_expr.scrutinee == scrutinee
        assert len(case_expr.arms) == 2
        assert case_expr.arms[0] == nil_arm
        assert case_expr.arms[1] == cons_arm
    
    def test_case_with_multiple_patterns(self):
        """Test case expression with multiple pattern types."""
        scrutinee = VarTerm("value")
        
        # case value of
        #   0 -> "zero"  # This would need literal patterns in full implementation
        #   x -> "other"
        #   _ -> "default"
        var_arm = CaseArm(VarPattern("x"), ConstructorTerm("String", []))
        wild_arm = CaseArm(WildcardPattern(), ConstructorTerm("Default", []))
        
        case_expr = CaseTerm(scrutinee, [var_arm, wild_arm])
        
        assert len(case_expr.arms) == 2
        assert isinstance(case_expr.arms[0].pattern, VarPattern)
        assert isinstance(case_expr.arms[1].pattern, WildcardPattern)
    
    def test_nested_case_expression(self):
        """Test nested case expressions."""
        # case x of
        #   Cons y ys -> case y of
        #                 Nil -> 1
        #                 _   -> 2
        #   Nil -> 0
        
        inner_nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(1))
        inner_wild_arm = CaseArm(WildcardPattern(), LitIntTerm(2))
        inner_case = CaseTerm(VarTerm("y"), [inner_nil_arm, inner_wild_arm])
        
        outer_cons_arm = CaseArm(
            ConstructorPattern("Cons", [VarPattern("y"), VarPattern("ys")]),
            inner_case
        )
        outer_nil_arm = CaseArm(ConstructorPattern("Nil", []), LitIntTerm(0))
        
        outer_case = CaseTerm(VarTerm("x"), [outer_cons_arm, outer_nil_arm])
        
        assert len(outer_case.arms) == 2
        assert isinstance(outer_case.arms[0].body, CaseTerm)
        assert isinstance(outer_case.arms[1].body, LitIntTerm)


class TestPatternComplexity:
    """Test complex pattern scenarios."""
    
    def test_deeply_nested_constructors(self):
        """Test deeply nested constructor patterns."""
        # Tree (Node (Leaf x) (Node (Leaf y) (Leaf z)))
        leaf_x = ConstructorPattern("Leaf", [VarPattern("x")])
        leaf_y = ConstructorPattern("Leaf", [VarPattern("y")])
        leaf_z = ConstructorPattern("Leaf", [VarPattern("z")])
        
        right_node = ConstructorPattern("Node", [leaf_y, leaf_z])
        root_node = ConstructorPattern("Node", [leaf_x, right_node])
        tree = ConstructorPattern("Tree", [root_node])
        
        expected = "Tree Node Leaf x Node Leaf y Leaf z"
        assert str(tree) == expected
    
    def test_pattern_with_many_variables(self):
        """Test pattern with many variables."""
        # Record a b c d e
        vars = [VarPattern(f"var_{i}") for i in range(10)]
        pattern = ConstructorPattern("Record", vars)
        
        expected_vars = " ".join(f"var_{i}" for i in range(10))
        expected = f"Record {expected_vars}"
        assert str(pattern) == expected
        assert len(pattern.args) == 10
    
    def test_mixed_pattern_types(self):
        """Test mixing all pattern types together."""
        # Complex x _ (Nested (Inner y) _) z
        inner = ConstructorPattern("Inner", [VarPattern("y")])
        nested = ConstructorPattern("Nested", [inner, WildcardPattern()])
        
        args = [
            VarPattern("x"),
            WildcardPattern(),
            nested,
            VarPattern("z")
        ]
        complex_pattern = ConstructorPattern("Complex", args)
        
        expected = "Complex x _ Nested Inner y _ z"
        assert str(complex_pattern) == expected


class TestPatternEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_constructor_args(self):
        """Test constructor with empty argument list."""
        pat = ConstructorPattern("Unit", [])
        assert pat.args == []
        assert str(pat) == "Unit"
    
    def test_single_arg_constructor(self):
        """Test constructor with single argument."""
        pat = ConstructorPattern("Just", [VarPattern("value")])
        assert len(pat.args) == 1
        assert str(pat) == "Just value"
    
    def test_case_with_no_arms(self):
        """Test case expression with no arms (unusual but possible)."""
        scrutinee = VarTerm("impossible")
        case_expr = CaseTerm(scrutinee, [])
        
        assert case_expr.scrutinee == scrutinee
        assert case_expr.arms == []
        assert len(case_expr.arms) == 0
    
    def test_case_with_single_arm(self):
        """Test case expression with single arm."""
        scrutinee = VarTerm("x")
        arm = CaseArm(WildcardPattern(), LitIntTerm(42))
        case_expr = CaseTerm(scrutinee, [arm])
        
        assert len(case_expr.arms) == 1
        assert case_expr.arms[0] == arm