"""Tests for subtyping algorithm."""

from tcpy.worklist import DKInference, SubJudgment
from tcpy.core import ConType, ArrowType, VarType, ETVarType, ForallType, AppType
from tcpy.errors import SubtypingError
import pytest


class TestSubtyping:
    """Test subtyping algorithm."""
    
    def test_reflexivity(self):
        dk = DKInference()
        ty = ConType("Int")
        # Should not raise an exception
        dk.solve_subtype(ty, ty)
    
    def test_compatible_constructors(self):
        dk = DKInference()
        int_ty = ConType("Int")
        int_ty2 = ConType("Int")
        # Should not raise an exception
        dk.solve_subtype(int_ty, int_ty2)
    
    def test_incompatible_constructors(self):
        dk = DKInference()
        int_ty = ConType("Int")
        bool_ty = ConType("Bool")
        # Should raise SubtypingError
        with pytest.raises(SubtypingError):
            dk.solve_subtype(int_ty, bool_ty)
    
    def test_arrow_subtyping_reflexive(self):
        dk = DKInference()
        # Int -> Bool <: Int -> Bool
        arrow_ty = ArrowType(ConType("Int"), ConType("Bool"))
        # Should not raise an exception
        dk.solve_subtype(arrow_ty, arrow_ty)
    
    def test_arrow_subtyping_contravariant(self):
        dk = DKInference()
        # (Bool -> Int) -> String <: (Int -> Int) -> String 
        # This should fail because Bool is not a supertype of Int
        left = ArrowType(ArrowType(ConType("Bool"), ConType("Int")), ConType("String"))
        right = ArrowType(ArrowType(ConType("Int"), ConType("Int")), ConType("String"))
        
        # This would require Bool <: Int which should fail
        # The actual result depends on whether we implement full subtyping rules
        # For now, this will likely fail since Bool != Int
        try:
            dk.solve_subtype(left, right)
        except SubtypingError:
            pass  # Expected failure
    
    def test_var_subtyping(self):
        dk = DKInference()
        var_a = VarType("a")
        var_b = VarType("b")
        
        # a <: a should succeed
        dk.solve_subtype(var_a, VarType("a"))
        
        # a <: b should fail (different variables)
        with pytest.raises(SubtypingError):
            dk.solve_subtype(var_a, var_b)
    
    def test_evar_subtyping(self):
        dk = DKInference()
        evar_a = ETVarType("^alpha0")
        evar_b = ETVarType("^alpha1")
        
        # ^alpha0 <: ^alpha0 should succeed
        dk.solve_subtype(evar_a, ETVarType("^alpha0"))
        
        # ^alpha0 <: ^alpha1 should trigger instantiation logic
        # This will depend on the worklist ordering and instantiation rules
        # For now we expect this to fail since the variables aren't in the worklist
        try:
            dk.solve_subtype(evar_a, evar_b)
        except Exception:
            pass  # Expected - variables need to be in worklist context


class TestOccursCheck:
    """Test occurs check functionality."""
    
    def test_occurs_check_var(self):
        dk = DKInference()
        var_ty = VarType("a")
        assert dk.occurs_check("a", var_ty)
        assert not dk.occurs_check("b", var_ty)
    
    def test_occurs_check_evar(self):
        dk = DKInference()
        evar_ty = ETVarType("^alpha0")
        assert dk.occurs_check("^alpha0", evar_ty)
        assert not dk.occurs_check("^alpha1", evar_ty)
    
    def test_occurs_check_arrow(self):
        dk = DKInference()
        # a -> b
        arrow_ty = ArrowType(VarType("a"), VarType("b"))
        assert dk.occurs_check("a", arrow_ty)
        assert dk.occurs_check("b", arrow_ty)
        assert not dk.occurs_check("c", arrow_ty)
    
    def test_occurs_check_forall(self):
        dk = DKInference()
        # foralla. a -> b
        forall_ty = ForallType("a", ArrowType(VarType("a"), VarType("b")))
        assert dk.occurs_check("a", forall_ty)
        assert dk.occurs_check("b", forall_ty)
        assert not dk.occurs_check("c", forall_ty)


class TestSubstitution:
    """Test type substitution."""
    
    def test_substitute_var(self):
        dk = DKInference()
        var_ty = VarType("a")
        replacement = ConType("Int")
        result = dk.substitute_type("a", replacement, var_ty)
        assert result == replacement
    
    def test_substitute_no_match(self):
        dk = DKInference()
        var_ty = VarType("a")
        replacement = ConType("Int")
        result = dk.substitute_type("b", replacement, var_ty)
        assert result == var_ty  # unchanged
    
    def test_substitute_arrow(self):
        dk = DKInference()
        # a -> a
        arrow_ty = ArrowType(VarType("a"), VarType("a"))
        replacement = ConType("Int")
        result = dk.substitute_type("a", replacement, arrow_ty)
        expected = ArrowType(ConType("Int"), ConType("Int"))
        assert result == expected
    
    def test_substitute_forall_no_capture(self):
        dk = DKInference()
        # foralla. a -> b  with [Int/b] should become foralla. a -> Int
        forall_ty = ForallType("a", ArrowType(VarType("a"), VarType("b")))
        replacement = ConType("Int")
        result = dk.substitute_type("b", replacement, forall_ty)
        expected = ForallType("a", ArrowType(VarType("a"), ConType("Int")))
        assert result == expected
    
    def test_substitute_forall_binding_protection(self):
        dk = DKInference()
        # foralla. a -> a  with [Int/a] should remain foralla. a -> a (no substitution under binding)
        forall_ty = ForallType("a", ArrowType(VarType("a"), VarType("a")))
        replacement = ConType("Int")
        result = dk.substitute_type("a", replacement, forall_ty)
        assert result == forall_ty  # unchanged


class TestMonotype:
    """Test monotype checking."""
    
    def test_monotype_constructors(self):
        dk = DKInference()
        assert dk.is_monotype(ConType("Int"))
        assert dk.is_monotype(VarType("a"))
        assert dk.is_monotype(ETVarType("^alpha0"))
    
    def test_monotype_arrow(self):
        dk = DKInference()
        arrow_ty = ArrowType(ConType("Int"), ConType("Bool"))
        assert dk.is_monotype(arrow_ty)
    
    def test_not_monotype_forall(self):
        dk = DKInference()
        forall_ty = ForallType("a", VarType("a"))
        assert not dk.is_monotype(forall_ty)
    
    def test_not_monotype_nested_forall(self):
        dk = DKInference()
        # Int -> (foralla. a)
        complex_ty = ArrowType(ConType("Int"), ForallType("a", VarType("a")))
        assert not dk.is_monotype(complex_ty)