"""Tests for DKInference basic structure."""

from tcpy.worklist import DKInference, ChkJudgment
from tcpy.core import ConType, LitIntTerm, VarTerm
from tcpy.errors import Ok


class TestDKInferenceBasic:
    """Test basic DKInference functionality."""
    
    def test_empty_inference_engine(self):
        dk = DKInference()
        assert len(dk.worklist.entries) == 0
        assert len(dk.trace) == 0
        assert len(dk.data_constructors) == 0
        assert len(dk.var_context) == 0
    
    def test_with_context_constructor(self):
        data_constructors = {"Nil": ConType("List")}
        var_context = {"x": ConType("Int")}
        
        dk = DKInference.with_context(data_constructors, var_context)
        assert dk.data_constructors == data_constructors
        assert dk.var_context == var_context
    
    def test_simple_check_type(self):
        dk = DKInference()
        term = LitIntTerm(42)
        expected_ty = ConType("Int")
        
        result = dk.check_type(term, expected_ty)
        assert isinstance(result, Ok)
    
    def test_trace_generation(self):
        dk = DKInference()
        term = LitIntTerm(42)
        expected_ty = ConType("Int")
        
        dk.check_type(term, expected_ty)
        trace = dk.get_trace()
        
        # Should have at least one trace entry
        assert len(trace) > 0
        assert "Chk 42 ⇐ Int" in trace
    
    def test_solve_empty_worklist(self):
        dk = DKInference()
        result = dk.solve()
        assert isinstance(result, Ok)
    
    def test_term_to_string(self):
        dk = DKInference()
        term = VarTerm("x")
        assert dk.term_to_string(term) == "x"
        
        term2 = LitIntTerm(42)
        assert dk.term_to_string(term2) == "42"