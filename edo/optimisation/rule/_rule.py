import operator
from copy import deepcopy

from .. import Goal, optimise
from ..categorisation import result_as_dict
from ... import make_origin
from ..._check import validate_index, assert_binary


class Rule(object):
    def __init__(self, origin, feature_index, class_index, class_name, action, goal,
                 criterion_relation, criterion_reference_value, name=None, derivation=None):
        """
        Rule defines how to modify Sample. It is implemented with the use of the following classes:
        - Action: defines action to be performed on feature values, e.g. removing or adding a substructure,
        - Goal: defines if the rule is intended to maximise or minimise the predicted value,
        - Criterion: an optional condition that must be satisfied to apply a rule,
                     it has the following form: shap_value relation reference_value, e.g. shap_value < 1,
        - Relation: defines relation used by Criterion.

        :param origin: Origin
            description of the model that provided SHAP values used to derive the rule
        :param feature_index: int
            index of the feature for which the rule is derived
        :param class_index: int or None
            in the case of classification, index of class for which the rule is derived, for regression should be None
        :param class_name: str or None
            in the case of classification, name of class for which the rule is derived (mapping between classes and
            their indices can be different for each model), for regression should be None
        :param action: Action
            defines action performed by the rule (e.g. removing a specific substructure)
        :param goal: Goal
            is the rule intended to maximise or minimise the predicted value
        :param criterion_relation: function (float,float) -> boolean
            a relation between the Sample's SHAP value and the criterion_reference_value, e.g. operator.lt
        :param criterion_reference_value: float
            value to which Sample's SHAP value is compared
        :param name: str or None
            human-friendly identifier of the Rule, if None then derived automatically; default: None
        :param derivation: SeparationResult, HighImpactResult, UnimportantResult, RandomRule or None
            determines in what way the rule was derived; default: None
        """
        validate_index(feature_index)
        validate_index(class_index)
        assert isinstance(goal, Goal)

        self.origin = make_origin(origin)
        self.ftr_idx = feature_index
        self.cls_idx = class_index
        self.cls_name = class_name

        self.goal = goal
        self.action = Action(action, self.ftr_idx)
        self.criterion = Criterion(self.origin, criterion_relation, criterion_reference_value,
                                   self.ftr_idx, self.cls_idx)

        self.name = f"F{self.ftr_idx}:{self.cls_name}: if {self.criterion} then {self.action} ({self.origin}) ({self.goal})" if name is None else name
        self.derivation = derivation if derivation is None else (derivation, self.goal)

    def apply(self, sample, skip_criterion_check=False, shap_calculator=None, extended_history=False):
        """
        Call edo.optimisation.optimise
        :param sample: Sample: sample to modify
        :param skip_criterion_check: boolean: if True then rule can be applied even if criterion is not satisfied;
                                              default: False
        :param shap_calculator: SHAPCalculator or None: a model to calculate SHAP values of the optimised sample,
                                if SHAP values should not be recalculated use None; default: None
        :param extended_history: if True will use ExtendedHistoryRecord instead of HistoryRecord; default: False
        :return: Sample: self
        """
        optimise(sample, self, skip_criterion_check, shap_calculator, extended_history)

    def can_be_applied(self, sample, skip_criterion_check=False):
        """
        Rule can be applied on Sample if all of the following are satisfied:
        - Rule and Sample share representation
        - Action is possible (based on feature values of Sample)
        - Criterion is satisfied (based on SHAP values of Sample) or skip_criterion_check=True

        :param sample: Sample: sample to check
        :param skip_criterion_check: boolean: if True then the rule can be applied even if self.criterion is not
                                              satisfied; default: False
        :return: bool: True if the rule can be applied, otherwise False
        """
        return (self.origin.representation == sample.origin.representation) \
               and self.action.is_possible(sample) \
               and (skip_criterion_check or self.criterion.is_satisfied(sample))

    def __str__(self):
        return f"If {self.criterion} then {self.action} ({self.goal})."

    def __repr__(self):
        return f"Rule({repr(self.origin)}, {repr(self.ftr_idx)}, {repr(self.cls_idx)}, {repr(self.cls_name)}, {repr(self.action)}, {repr(self.goal)}, {repr(self.criterion.relation)}, {repr(self.criterion.relation.ref_val)}, name={repr(self.name)}, derivation={repr(self.derivation) if self.derivation is None else repr(self.derivation[0])})"

    def as_dict(self, compact=False):
        """
        Human-friendly version of self._asdict().
        :param compact: boolean: if True only a short description of derivation is included; default: False
        :return: dictionary describing the object
        """
        d = deepcopy(self.__dict__)
        d['goal'] = d['goal'].name
        d['action'] = d['action'].desc
        d['criterion_relation'] = self.criterion.relation.relation.__qualname__
        d['criterion_ref_val'] = self.criterion.relation.ref_val
        del d['criterion']
        del d['name']
        d.update(d['origin']._asdict())
        del d['origin']
        if self.derivation is not None:
            d['derivation'] = self.derivation_as_dict()
            d['derivation_type'] = d['derivation']['type']
            d['derivation_score'] = d['derivation']['score']
            if compact:
                del d['derivation']
        return d

    def derivation_as_dict(self):
        """
        Human-friendly version of self.derivation._asdict().
        :return: dictionary describing the derivation
        """
        if self.derivation is None:
            return {}
        d = result_as_dict(self.derivation[0])
        d['goal'] = self.derivation[1].name
        return d

    def contradicts(self, other):
        # origin and feature index must match for comparison to be possible
        # only action and goal are checked to calculate if rules are contradictory

        # Can we compare these rules?
        assert self.origin == other.origin, f"Origin mismatch {self.origin} != {other.origin}"
        assert self.ftr_idx == other.ftr_idx, f"Feature index mismatch {self.ftr_idx} != {other.ftr_idx}"

        if self.goal == other.goal and self.action.desc == other.action.desc:
            return False
        elif self.goal != other.goal and self.action.desc != other.action.desc:
            # NOTE: this assumes that there are exactly two possible goals and exactly two possible actions
            return False
        else:
            # same goals but different actions or same actions but different goals
            return True


# # # # # # # # # # # # # #
# F A I R   W A R N I N G #
# You don't need the stuff below. Just use Sample, Rule and edo.optimisation.optimise().
# # # # # # # # # # # # # #

class Criterion(object):
    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        """
        Criterion which Sample's SHAP values must satisfy for Rule to be applied. The condition has the following form:
        shap_value relation reference_value, e.g: shap_value < 1.

        origin: Origin
            inf. about the model for which Criterion is defined
            used to check if Sample's SHAP values are derived for the same model
        relation: function (float,float) -> boolean
            required relation of the Sample's SHAP value to the reference_value,
            preferably from the operator module, ex. operator.lt
        reference_value: float
            value to which Sample's SHAP value is compared
        feature_index: int
            index of feature for which Criterion is defined
        class_index: int or None
            in the case of classification index of class for which Criterion is defined,
            in the case of regression should be None
        """
        self.origin = make_origin(origin)
        self.relation = make_relation(relation, reference_value)
        self.ftr_idx = feature_index
        self.cls_idx = class_index

    def is_satisfied(self, sample):
        assert self.origin == sample.s_vals_origin, f"Cannot check criterion for SHAP values calculated for a different origin! Criterion origin is {self.origin} != {sample.origin} (SHAP values origin)."

        if self.cls_idx is None:
            s_vals = sample.s_vals[self.ftr_idx]
        else:
            s_vals = sample.s_vals[self.cls_idx, self.ftr_idx]
        return self.relation(s_vals)

    def __str__(self):
        cls_idx = f"[{self.cls_idx}]" if self.cls_idx is not None else ''
        return self.relation.__str__(f"shap_val{cls_idx}[{self.ftr_idx}]")

    def __repr__(self):
        return f"Criterion({repr(self.origin)}, {repr(self.relation.relation)}, {repr(self.relation.ref_val)}, {repr(self.ftr_idx)}, {repr(self.cls_idx)})"


class Action(object):
    def __init__(self, func, index):
        """
        Action to perform on Sample

        Action can be adding or removing a substructure (or feature in general).
        Only binary representations are supported.

        func: str
           'add' to add a substructure (set value to 1), 'remove' to remove a substructure (set value to 0)
           notice: if more general Actions are required, then functions should be used instead of strings
                   and implementation would need a small update as well
        index: int
            index of feature on which Action is performed
        """
        self.index = index
        func = func.lower()
        assert func in ['add', 'remove'], f'`func` must be `add` or `remove`, is {func}.'
        self.desc = func  # human-readable format
        self.new_value = 1 if func == 'add' else 0

    def do(self, vector):
        # we want to avoid changing Sample (or anything else) inplace bc this prevents from tracking changes
        # copying could be skipped (it's already done in Sample.__init__) but better (double) safe than sorry
        # so if a user calls this method directly (not via optimise function) then we are still safe
        new_vec = deepcopy(vector)
        new_vec[self.index] = self.new_value
        return new_vec

    def is_possible(self, sample):
        assert_binary(sample.f_vals[self.index])
        # can add substructure only if it is not there yet...
        return sample.f_vals[self.index] != self.new_value

    def __str__(self):
        return f"f_val[{self.index}] -> {self.new_value}"

    def __repr__(self):
        return f"Action({repr(self.desc)}, {repr(self.index)})"

    def equals(self, other):
        # define own comparator method but don't overwrite __eq__
        return (self.index == other.index) and (self.new_value == other.new_value)


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
