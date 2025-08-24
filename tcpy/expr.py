from dataclasses import dataclass
from tcpy.type import Type

class Expr:
    pass

@dataclass
class Var(Expr):
    id: str

@dataclass
class App(Expr):
    arg: Expr
    expr: Expr

@dataclass
class Abs(Expr):
    id: str
    ty: Type
    expr: Expr

@dataclass
class TApp(Expr):
    expr: Expr
    ty: Type

@dataclass
class TAbs(Expr):
    id: str
    expr: Expr

@dataclass
class Ann(Expr):
    expr: Expr
    ty: Type

@dataclass
class LitInt(Expr):
    val: int

@dataclass
class LitBool(Expr):
    val: bool

@dataclass
class Let(Expr):
    id: str
    expr1: Expr
    expr2: Expr

@dataclass
class IfThenElse(Expr):
    expr1: Expr
    expr2: Expr
    expr3: Expr

@dataclass
class BinOp(Expr):
    op: str
    expr1: Expr
    expr2: Expr
