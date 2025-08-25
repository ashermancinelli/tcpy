"""Binary operations for System-F-omega terms."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


class CoreBinOp(ABC):
    """Binary operations."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class AddOp(CoreBinOp):
    def __str__(self) -> str:
        return "+"


@dataclass
class SubOp(CoreBinOp):
    def __str__(self) -> str:
        return "-"


@dataclass
class MulOp(CoreBinOp):
    def __str__(self) -> str:
        return "*"


@dataclass
class DivOp(CoreBinOp):
    def __str__(self) -> str:
        return "/"


@dataclass
class LtOp(CoreBinOp):
    def __str__(self) -> str:
        return "<"


@dataclass
class LeOp(CoreBinOp):
    def __str__(self) -> str:
        return "<="