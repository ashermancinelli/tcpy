"""Tests for exception-based error handling."""

from tcpy.worklist import DKInference
from tcpy.core import ConType, VarTerm, LitIntTerm
from tcpy.errors import UnboundVariableError
import pytest


class TestExceptionHandling:
    """Test exception-based error handling."""
    
    def test_successful_type_check(self):
        dk = DKInference()
        term = LitIntTerm(42)
        expected_ty = ConType("Int")
        
        # Should not raise an exception
        dk.check_type(term, expected_ty)
        
        # Check that trace was generated
        trace = dk.get_trace()
        assert len(trace) > 0
        assert any("Chk 42 <= Int" in entry for entry in trace)
    
    def test_unbound_variable_error(self):
        dk = DKInference()
        term = VarTerm("x")
        expected_ty = ConType("Int")
        
        # Should raise UnboundVariableError
        with pytest.raises(UnboundVariableError) as exc_info:
            dk.check_type(term, expected_ty)
        
        assert exc_info.value.name == "x"
    
    def test_bound_variable_success(self):
        dk = DKInference()
        dk.var_context = {"x": ConType("Int")}
        
        term = VarTerm("x")
        expected_ty = ConType("Int")
        
        # Should not raise an exception
        dk.check_type(term, expected_ty)