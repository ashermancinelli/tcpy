"""Error types for the type checker."""

from typing import Optional, Tuple
from dataclasses import dataclass

# Type alias for source location spans
Span = Optional[Tuple[int, int]]


class TypeError(Exception):
    """Base class for type checking errors."""
    def __init__(self, message: str, span: Span = None):
        self.message = message
        self.span = span
        super().__init__(message)


@dataclass
class UnboundVariableError(TypeError):
    """Variable not found in scope."""
    name: str
    span: Span = None
    
    def __init__(self, name: str, span: Span = None):
        self.name = name
        self.span = span
        super().__init__(f"Variable '{name}' not found in scope", span)


@dataclass  
class UnboundDataConstructorError(TypeError):
    """Data constructor not found."""
    name: str
    span: Span = None
    
    def __init__(self, name: str, span: Span = None):
        self.name = name
        self.span = span
        super().__init__(f"Data constructor '{name}' not found", span)


@dataclass
class NotAFunctionError(TypeError):
    """Type is not a function type."""
    ty: 'CoreType'  # Forward reference
    span: Span = None
    
    def __init__(self, ty: 'CoreType', span: Span = None):
        self.ty = ty
        self.span = span
        super().__init__(f"Type '{ty}' is not a function type", span)


@dataclass
class ArityMismatchError(TypeError):
    """Wrong number of arguments."""
    expected: int
    actual: int
    span: Span = None
    
    def __init__(self, expected: int, actual: int, span: Span = None):
        self.expected = expected
        self.actual = actual
        self.span = span
        super().__init__(f"Arity mismatch: expected {expected} arguments, got {actual}", span)


@dataclass
class SubtypingError(TypeError):
    """Subtyping relationship failed."""
    left: 'CoreType'  # Forward reference
    right: 'CoreType'  # Forward reference
    span: Span = None
    
    def __init__(self, left: 'CoreType', right: 'CoreType', span: Span = None):
        self.left = left
        self.right = right
        self.span = span
        super().__init__(f"Subtyping failure: '{left}' is not a subtype of '{right}'", span)


@dataclass
class InstantiationError(TypeError):
    """Type variable instantiation failed."""
    var: str
    ty: 'CoreType'  # Forward reference
    span: Span = None
    
    def __init__(self, var: str, ty: 'CoreType', span: Span = None):
        self.var = var
        self.ty = ty
        self.span = span
        super().__init__(f"Instantiation failure: cannot instantiate '{var}' with '{ty}'", span)


# Remove old type aliases - no longer needed