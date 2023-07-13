import operator


def make_relation(relation, reference_value):
    if relation in RelationOp._allowed:
        return RelationOp(relation, reference_value)
    else:
        return Relation(relation, reference_value)


class Relation(object):
    # TODO: doksy!
    def __init__(self, relation, reference_value):
        self.relation = relation
        self.ref_val = reference_value

    def __str__(self, x=None):
        # ex. 'x < 3' (if x is None it should be overwritten with 'x')
        # TODO: ...
        return f"{'x' if x is None else x} {self.relation.__qualname__} {self.ref_val}"

    def __repr__(self):
        return f"Relation({repr(self.relation)}, {repr(self.ref_val)})"

    def __call__(self, x):
        return self.relation(x, self.ref_val)


class RelationOp(Relation):
    # supports operator.lt, operator.gt, operator.le, operator.ge
    def __init__(self, relation, reference_value):
       assert relation in RelationOp._allowed, f"{relation} is not in {RelationOp._allowed}."
       super().__init__(relation, reference_value)

    _allowed = [operator.lt, operator.gt, operator.le, operator.ge]
    _relnames = {'lt': '<', 'le': '≤', 'gt': '>', 'ge': '≥'}  # for pretty printing

    def __str__(self, x=None):
        x = 'x' if x is None else x
        return f"{x} {RelationOp._relnames[self.relation.__name__]} {self.ref_val}"

    def __repr__(self, ):
        return f"RelationOp({repr(self.relation)}, {repr(self.ref_val)})"
