from dataclasses import dataclass, field

class Kind:
		pass

class Star(Kind):
		pass

class KArrow(Kind):
		domain: Kind
		codomain: Kind

class Type:
    pass

@dataclass
class Var(Type):
    '''Regular type variable'''
    id: str

@dataclass
class ETVar(Type):
    '''Existential type variable'''
    id: str

@dataclass
class Oper(Type):
    '''Type operator'''
    id: str
    subtypes: list[Type] = field(default_factory=list)

@dataclass
class Forall(Type):
    etvar: str
    subtype: Type
