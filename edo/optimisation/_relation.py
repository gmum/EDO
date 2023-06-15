import operator
from collections.abc import Iterable
from abc import abstractmethod
import numpy as np


# To jest makabrycznie rozbuchana abstrakcja, pozwalająca na łatwe sprawdzenie, czy dwa przedziały mają puste przecięcie


def make_relation(relation, reference_value):
    if relation in RelationOp._allowed:
        return RelationOpSingle(relation, reference_value)
    elif isinstance(relation, Iterable) and np.all(r in RelationOp._allowed for r in relation):
        return RelationOpDouble(relation, reference_value)
    else:
        return Relation(relation, reference_value)


class Relation(object):
    # TODO: doksy!
    def __init__(self, relation, reference_value):
        self.relation = relation
        self.ref_val = reference_value

    @abstractmethod
    def to_interval(self, ):
        # should return left_value, left_inclusive, right_value, right_inclusive
        pass

    # @abstractmethod
    def __str__(self, x=None):
        # ex. 'x < 3' (if x is None it should be overwritten with 'x')
        # TODO: ta funkcja powinna być pusta? To jest abstract method. Dlaczego mogę to wołać?
        return f"{'x' if x is None else x} {self.relation.__qualname__} {self.ref_val}"
        pass

    def __repr__(self, ):
        return f"Relation({repr(self.relation)}, {repr(self.ref_val)})"

    def __call__(self, x):
        return self.relation(x, self.ref_val)

    def overlaps(self, other):
        # TODO: nietestowane, ale też nie będzie używane...
        s_l_val, s_l_inclusive, s_r_val, s_r_inclusive = self.to_interval()
        o_l_val, o_l_inclusive, o_r_val, o_r_inclusive = other.to_interval()

        # "sorting" intervals
        # we want intervals /a, b/ and /c, d/ such that that b <= d
        if s_r_val <= o_r_inclusive:
            b = s_r_val
            c = o_l_val
            b_incl = s_r_inclusive
            c_incl = o_l_inclusive
        else:
            b = o_r_val
            c = s_l_val
            b_incl = o_r_inclusive
            c_incl = s_l_inclusive

        if c > b:
            # a-----b
            #          c-----d
            return False
        elif c < b:
            # a-----b
            #    c-----d
            return True
        else:
            # a-----b
            #       c-----d
            # if both are inclusive then there is overlap
            return (b_incl and c_incl)


class RelationOp(Relation):
    # supports operator.lt, operator.gt, operator.le, operator.ge
    def __init__(self, relation, reference_value):
        try:
            assert relation in RelationOp._allowed, f"{relation} is not in {RelationOp._allowed}."
        except AssertionError:
            assert np.all(r in RelationOp._allowed for r in relation), f"{relation} is not in {RelationOp._allowed}."
        super().__init__(relation, reference_value)

    _allowed = [operator.lt, operator.gt, operator.le, operator.ge]
    _mutually_inclusive = [operator.le, operator.ge]
    _pointing_left = [operator.le, operator.lt]
    _relnames = {'lt': '<', 'le': '≤', 'gt': '>', 'ge': '≥'}  # for pretty printing


class RelationOpSingle(RelationOp):
    def __init__(self, relation, reference_value):
        super().__init__(relation, reference_value)

    def to_interval(self, ):
        # TODO: nietestowane, ale też nie będzie używane...
        if self.relation in RelationOp._pointing_left:
            l_val = -np.inf
            l_inclusive = False
            r_val = self.ref_val
            r_inclusive = self.relation in RelationOp._mutually_inclusive
        else:
            l_val = self.ref_val
            l_inclusive = self.relation in RelationOp._mutually_inclusive
            r_val = np.inf
            r_inclusive = False

        return l_val, l_inclusive, r_val, r_inclusive

    def __str__(self, x=None):
        x = 'x' if x is None else x
        return f"{x} {RelationOp._relnames[self.relation.__name__]} {self.ref_val}"

    def __repr__(self, ):
        return f"RelationOpSingle({repr(self.relation)}, {repr(self.ref_val)})"


class RelationOpDouble(RelationOp):
    # left_reference_value left_relation x right_relation right_reference_value, ex. 2 < x <= 3
    def __init__(self, relation, reference_value):
        assert len(relation) == 2, f"Exactly two relations are required, given {len(relation)}."
        assert len(reference_value) == 2, f"Exactly two reference values are required, given {len(reference_value)}."
        assert reference_value[0] <= reference_value[
            1], f"Required that reference_value[0] <= reference_value[1], given {reference_value}."
        assert (relation[0] in RelationOp._pointing_left) == (relation[
                                                                  1] in RelationOp._pointing_left), f"Relations must point in the same direction, given {relation}."

        super().__init__(relation, reference_value)
        self.l_relation = self.relation[0]
        self.r_relation = self.relation[1]
        self.l_ref_val = self.ref_val[0]
        self.r_ref_val = self.ref_val[1]

        raise NotImplementedError("__call__ dla RelationOpDouble nie zadziała")

    def to_interval(self, ):
        # TODO: nietestowane, ale też nie będzie używane...
        l_val = self.l_ref_val
        l_inclusive = self.l_relation in RelationOp._mutually_inclusive
        r_val = self.r_ref_val
        r_inclusive = self.r_relation in RelationOp._mutually_inclusive

        return l_val, l_inclusive, r_val, r_inclusive

    def __str__(self, x=None):
        x = 'x' if x is None else x
        return f"{self.l_ref_val} {RelationOp._relnames[self.l_relation.__name__]} {x} {RelationOp._relnames[self.r_relation.__name__]} {self.r_ref_val}"

    def __repr__(self, ):
        return f"RelationOpDouble({repr(self.relation)}, {repr(self.ref_val)})"
