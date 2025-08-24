"""Tests for basic worklist functionality."""

import pytest
from tcpy.worklist import (
    Worklist, WorklistEntry, TyVarEntry, VarEntry, JudgmentEntry,
    TyVarKind, UniversalTyVar, ExistentialTyVar, SolvedTyVar, MarkerTyVar,
    Judgment, SubJudgment, InfJudgment, ChkJudgment, InfAppJudgment
)
from tcpy.core import ConType, VarTerm, LitIntTerm
from tcpy.errors import Ok, Err, UnboundVariableError


class TestTyVarKinds:
    """Test type variable kinds."""
    
    def test_universal_tyvar(self):
        kind = UniversalTyVar()
        assert isinstance(kind, TyVarKind)
    
    def test_existential_tyvar(self):
        kind = ExistentialTyVar()
        assert isinstance(kind, TyVarKind)
    
    def test_solved_tyvar(self):
        ty = ConType("Int")
        kind = SolvedTyVar(ty)
        assert isinstance(kind, TyVarKind)
        assert kind.solution == ty
    
    def test_marker_tyvar(self):
        kind = MarkerTyVar()
        assert isinstance(kind, TyVarKind)


class TestJudgments:
    """Test judgment types."""
    
    def test_sub_judgment(self):
        left = ConType("Int")
        right = ConType("Bool")
        judgment = SubJudgment(left, right)
        assert isinstance(judgment, Judgment)
        assert judgment.left == left
        assert judgment.right == right
    
    def test_inf_judgment(self):
        term = VarTerm("x")
        ty = ConType("Int")
        judgment = InfJudgment(term, ty)
        assert isinstance(judgment, Judgment)
        assert judgment.term == term
        assert judgment.ty == ty
    
    def test_chk_judgment(self):
        term = LitIntTerm(42)
        ty = ConType("Int")
        judgment = ChkJudgment(term, ty)
        assert isinstance(judgment, Judgment)
        assert judgment.term == term
        assert judgment.ty == ty


class TestWorklistEntries:
    """Test worklist entry types."""
    
    def test_tyvar_entry(self):
        name = "α0"
        kind = UniversalTyVar()
        entry = TyVarEntry(name, kind)
        assert isinstance(entry, WorklistEntry)
        assert entry.name == name
        assert entry.kind == kind
    
    def test_var_entry(self):
        name = "x"
        ty = ConType("Int")
        entry = VarEntry(name, ty)
        assert isinstance(entry, WorklistEntry)
        assert entry.name == name
        assert entry.ty == ty
    
    def test_judgment_entry(self):
        judgment = SubJudgment(ConType("Int"), ConType("Bool"))
        entry = JudgmentEntry(judgment)
        assert isinstance(entry, WorklistEntry)
        assert entry.judgment == judgment


class TestWorklist:
    """Test Worklist class."""
    
    def test_empty_worklist(self):
        wl = Worklist()
        assert len(wl.entries) == 0
        assert wl.next_var == 0
        assert wl.pop() is None
    
    def test_fresh_var_generation(self):
        wl = Worklist()
        var1 = wl.fresh_var()
        var2 = wl.fresh_var()
        assert var1 == "α0"
        assert var2 == "α1"
        assert var1 != var2
    
    def test_fresh_evar_generation(self):
        wl = Worklist()
        evar1 = wl.fresh_evar()
        evar2 = wl.fresh_evar()
        assert evar1 == "^α0"
        assert evar2 == "^α1"
        assert evar1 != evar2
    
    def test_push_pop(self):
        wl = Worklist()
        entry = TyVarEntry("α0", UniversalTyVar())
        
        wl.push(entry)
        assert len(wl.entries) == 1
        
        popped = wl.pop()
        assert popped == entry
        assert len(wl.entries) == 0
    
    def test_find_var_success(self):
        wl = Worklist()
        ty = ConType("Int")
        entry = VarEntry("x", ty)
        wl.push(entry)
        
        found_ty = wl.find_var("x")
        assert found_ty == ty
    
    def test_find_var_not_found(self):
        wl = Worklist()
        found_ty = wl.find_var("x")
        assert found_ty is None
    
    def test_find_var_latest_binding(self):
        wl = Worklist()
        ty1 = ConType("Int")
        ty2 = ConType("Bool")
        
        wl.push(VarEntry("x", ty1))
        wl.push(VarEntry("x", ty2))
        
        # Should find the most recent binding
        found_ty = wl.find_var("x")
        assert found_ty == ty2
    
    def test_solve_evar_success(self):
        wl = Worklist()
        evar_name = "^α0"
        entry = TyVarEntry(evar_name, ExistentialTyVar())
        wl.push(entry)
        
        solution = ConType("Int")
        result = wl.solve_evar(evar_name, solution)
        
        assert isinstance(result, Ok)
        assert isinstance(entry.kind, SolvedTyVar)
        assert entry.kind.solution == solution
    
    def test_solve_evar_not_found(self):
        wl = Worklist()
        result = wl.solve_evar("^α0", ConType("Int"))
        
        assert isinstance(result, Err)
        assert isinstance(result.error, UnboundVariableError)
        assert result.error.name == "^α0"
    
    def test_solve_evar_already_solved(self):
        wl = Worklist()
        evar_name = "^α0"
        original_solution = ConType("Int")
        entry = TyVarEntry(evar_name, SolvedTyVar(original_solution))
        wl.push(entry)
        
        new_solution = ConType("Bool")
        result = wl.solve_evar(evar_name, new_solution)
        
        # Should succeed but not change the solution
        assert isinstance(result, Ok)
        assert isinstance(entry.kind, SolvedTyVar)
        assert entry.kind.solution == original_solution  # unchanged
    
    def test_before_ordering(self):
        wl = Worklist()
        wl.push(TyVarEntry("α0", UniversalTyVar()))
        wl.push(TyVarEntry("α1", ExistentialTyVar()))
        
        assert wl.before("α0", "α1")
        assert not wl.before("α1", "α0")
    
    def test_before_not_found(self):
        wl = Worklist()
        wl.push(TyVarEntry("α0", UniversalTyVar()))
        
        assert not wl.before("α0", "α1")  # α1 not found
        assert not wl.before("α1", "α0")  # α1 not found
        assert not wl.before("α1", "α2")  # neither found