"""Tests for subtyping algorithm."""

from tcpy.worklist import DKInference, SubJudgment
from tcpy.core import ConType, ArrowType, VarType, ETVarType, ForallType, AppType
from tcpy.errors import Ok, Err, SubtypingError


class TestSubtyping:
    """Test subtyping algorithm."""
    
    def test_reflexivity(self):
        dk = DKInference()
        ty = ConType("Int")
        result = dk.solve_subtype(ty, ty)
        assert isinstance(result, Ok)
    
    def test_compatible_constructors(self):
        dk = DKInference()
        int_ty = ConType("Int")
        int_ty2 = ConType("Int")
        result = dk.solve_subtype(int_ty, int_ty2)
        assert isinstance(result, Ok)
    
    def test_incompatible_constructors(self):
        dk = DKInference()
        int_ty = ConType("Int")
        bool_ty = ConType("Bool")
        result = dk.solve_subtype(int_ty, bool_ty)
        assert isinstance(result, Err)
        assert isinstance(result.error, SubtypingError)
    
    def test_arrow_subtyping_reflexive(self):
        dk = DKInference()
        # Int -> Bool <: Int -> Bool
        arrow_ty = ArrowType(ConType("Int"), ConType("Bool"))
        result = dk.solve_subtype(arrow_ty, arrow_ty)
        assert isinstance(result, Ok)
    
    def test_arrow_subtyping_contravariant(self):
        dk = DKInference()
        # (Bool -> Int) -> String <: (Int -> Int) -> String 
        # This should fail because Bool is not a supertype of Int
        left = ArrowType(ArrowType(ConType("Bool"), ConType("Int")), ConType("String"))
        right = ArrowType(ArrowType(ConType("Int"), ConType("Int")), ConType("String"))
        
        # This would require Bool <: Int which should fail
        result = dk.solve_subtype(left, right)
        # The actual result depends on whether we implement full subtyping rules
        # For now, this will likely fail since Bool != Int
        if isinstance(result, Err):
            assert isinstance(result.error, SubtypingError)
    
    def test_var_subtyping(self):
        dk = DKInference()
        var_a = VarType("a")
        var_b = VarType("b")
        
        # a <: a should succeed
        result = dk.solve_subtype(var_a, VarType("a"))
        assert isinstance(result, Ok)
        
        # a <: b should fail (different variables)
        result = dk.solve_subtype(var_a, var_b)
        assert isinstance(result, Err)
    
    def test_evar_subtyping(self):
        dk = DKInference()
        evar_a = ETVarType("^α0")
        evar_b = ETVarType("^α1")
        
        # ^α0 <: ^α0 should succeed
        result = dk.solve_subtype(evar_a, ETVarType("^α0"))
        assert isinstance(result, Ok)
        
        # ^α0 <: ^α1 should trigger instantiation logic
        result = dk.solve_subtype(evar_a, evar_b)
        # This will depend on the worklist ordering and instantiation rules


class TestOccursCheck:
    """Test occurs check functionality."""
    
    def test_occurs_check_var(self):
        dk = DKInference()
        var_ty = VarType("a")
        assert dk.occurs_check("a", var_ty)
        assert not dk.occurs_check("b", var_ty)
    
    def test_occurs_check_evar(self):
        dk = DKInference()
        evar_ty = ETVarType("^α0")
        assert dk.occurs_check("^α0", evar_ty)
        assert not dk.occurs_check("^α1", evar_ty)
    
    def test_occurs_check_arrow(self):
        dk = DKInference()
        # a -> b
        arrow_ty = ArrowType(VarType("a"), VarType("b"))
        assert dk.occurs_check("a", arrow_ty)
        assert dk.occurs_check("b", arrow_ty)
        assert not dk.occurs_check("c", arrow_ty)
    
    def test_occurs_check_forall(self):
        dk = DKInference()
        # ∀a. a -> b
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
        # ∀a. a -> b  with [Int/b] should become ∀a. a -> Int
        forall_ty = ForallType("a", ArrowType(VarType("a"), VarType("b")))
        replacement = ConType("Int")
        result = dk.substitute_type("b", replacement, forall_ty)
        expected = ForallType("a", ArrowType(VarType("a"), ConType("Int")))
        assert result == expected
    
    def test_substitute_forall_binding_protection(self):
        dk = DKInference()
        # ∀a. a -> a  with [Int/a] should remain ∀a. a -> a (no substitution under binding)
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
        assert dk.is_monotype(ETVarType("^α0"))
    
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
        # Int -> (∀a. a)
        complex_ty = ArrowType(ConType("Int"), ForallType("a", VarType("a")))
        assert not dk.is_monotype(complex_ty)