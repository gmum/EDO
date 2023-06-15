from enum import Enum


# TODO: should be renamed, we have relation in src.optimisation
class Relation(Enum):
    MOST = 'most'
    LEAST = 'least'


# TODO: has to be renamed!!! Category -> Stability?
class Category(Enum):
    UNSTABLE = 0
    MEDIUM = 1
    STABLE = 2


