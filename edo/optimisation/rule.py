from copy import deepcopy

from . import Goal, optimise
from ._relation import make_relation
from .. import make_origin
from .._check import validate_index, assert_binary
from ..shap_analysis.categorisation import result_as_dict


class Rule(object):
    """
    Rule: defines how to change Sample

    Rule can be applied on Sample if all of the following are satisfied:
    - Rule and Sample share origin (or at least representation of the original model)
    - Action is possible (based on feature values of Sample)
    - Criterion is satisfied (based on SHAP values of Sample)

    Action can be removing or adding a substructure.
    Criterion has the following form: shap_value relation reference_value, ex: shap_value < 1

    origin: Origin
        inf. about the model for which the rule is derived
    feature_index: int
        index of feature for which the rule is derived
    class_index: int or None
        used to check if Criterion is satisfied
        in the case of classification index of class for which Rule is derived,
        in the case of regression should be None
    class_name: str or None
        used to check if two Rules are contradictory
        in the case of classification name of class for which Rule is derived
        (mapping between classes and their indices can be different for each model)
        in the case of regression should be None
    action: Action
        defines action performed by Rule (ex. removing specific substructure)
    goal: Goal
        does this rule maximise or minimise the predicted value?
    criterion_relation: function (float,float) -> boolean
        required relation of the Sample's SHAP value to the criterion_reference_value,
        preferably from operator module, ex. operator.lt
    criterion_reference_value: float
        value to which Sample's SHAP value is compared
    name: str or None
        human-friendly identifier of the Rule, if None then derived automatically; default: None
    derivation: anything really
        If two Rules contradict one another I will need this to determine what happened
    """

    def __init__(self, origin, feature_index, class_index, class_name, action, goal,
                 criterion_relation, criterion_reference_point, name=None, derivation=None):
        validate_index(feature_index)
        validate_index(class_index)
        assert isinstance(goal, Goal)

        self.origin = make_origin(origin)
        self.ftr_idx = feature_index
        # class_index and class_name might have different mappings for different models
        # we need index to apply a rule,
        # we need name to check if rules do not contradict one another
        self.cls_idx = class_index
        self.cls_name = class_name

        self.goal = goal
        self.action = Action(action, self.ftr_idx)
        self.criterion = Criterion(self.origin, criterion_relation, criterion_reference_point,
                                   self.ftr_idx, self.cls_idx)

        self.name = f"F{self.ftr_idx}:{self.cls_name}: if {self.criterion} then {self.action} ({self.origin}) ({self.goal})" if name is None else name
        # derivation -> justification?
        self.derivation = derivation  # ex. Well separated (regions: ...; majorities: ..)

    def apply(self, sample, skip_criterion_check=False, update_shap=None, extended_history=False):
        # call optimise
        optimise(sample, self, skip_criterion_check, update_shap, extended_history)

    def can_be_applied(self, sample, skip_criterion_check=False):
        return self.action.is_possible(sample) and (skip_criterion_check or self.criterion.is_satisfied(sample))

    def __str__(self, ):
        return f"If {self.criterion} then {self.action} ({self.goal})."

    def __repr__(self, ):
        return f"Rule(origin={repr(self.origin)}, feature_index={repr(self.ftr_idx)}, class_index={repr(self.cls_idx)}, class_name={repr(self.cls_name)}, action={repr(self.action)}, goal={repr(self.goal)}, criterion_relation={repr(self.criterion.relation)}, criterion_reference_point={repr(self.criterion.relation.ref_val)}, name={repr(self.name)}, derivation={repr(self.derivation)})"

    def as_dict(self, compact=False):
        d = deepcopy(self.__dict__)
        d['goal'] = d['goal'].name
        d['action'] = d['action'].desc
        d['criterion_relation'] = self.criterion.relation.relation.__qualname__
        d['criterion_ref_val'] = self.criterion.relation.ref_val
        del d['criterion']
        del d['name']
        d.update(d['origin']._asdict())
        del d['origin']
        d['derivation'] = self.derivation_as_dict()
        d['derivation_type'] = d['derivation']['type']
        d['derivation_score'] = d['derivation']['score']
        if compact:
            del d['derivation']
        return d

    def derivation_as_dict(self):
        d = result_as_dict(self.derivation[0])
        d['goal'] = self.derivation[1].name
        return d

    def contradicts(self, other):
        # origin and feature index must match for comparison to be possible
        # only action and goal are checked to calculate if rules are contradictive
        # old contradict_simple from mock_experiment

        # Can we compare these rules?
        # TODO: potencjalnie moglibysmy chciec porownywac reguły wygenerowane dla różnych modeli
        # wtedy zamiast sprawdzać równość całego origin trzebaby sprwadzać tylko origin.representation
        assert self.origin == other.origin, f"Origin mismatch {self.origin} != {other.origin}"
        assert self.ftr_idx == other.ftr_idx, f"Fature index mismatch {self.ftr_idx} != {other.ftr_idx}"

        if self.goal == other.goal and self.action.desc == other.action.desc:
            return False
        elif self.goal != other.goal and self.action.desc != other.action.desc:
            # TODO! uwaga: założyliśmy, że możliwe są dokładnie dwa cele i dokładnie dwie akcje
            # jak zmienimy api, to tutaj będzie trzeba naprawić
            return False
        else:
            # te same cele i różne akcje albo te same akcje i różne cele
            return True


# # # F A I R   W A R N I N G # # #
# You don't need the stuff below.
# Just use Sample, Rule and optimise().
class Criterion(object):
    """
    Criterion which Sample's SHAP values must satisfy for Rule to be applied

    Criterion has the following form: shap_value relation reference_value, ex: shap_value < 1

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

    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        self.origin = make_origin(origin)
        self.relation = make_relation(relation, reference_value)
        self.ftr_idx = feature_index
        self.cls_idx = class_index

    def is_satisfied(self, sample):
        assert self.origin == sample.s_vals_origin, f"Cannot check criterion for SHAP values calculated for a different origin! Criterion origin is {self.origin} != {sample.s_vals_origin} (SHAP values origin)."

        if self.cls_idx is None:
            s_vals = sample.s_vals[self.ftr_idx]
        else:
            s_vals = sample.s_vals[self.cls_idx, self.ftr_idx]
        return self.relation(s_vals)

    def __str__(self, ):
        cls_idx = f"[{self.cls_idx}]" if self.cls_idx is not None else ''
        return self.relation.__str__(f"shap_val{cls_idx}[{self.ftr_idx}]")

    def __repr__(self, ):
        return f"Criterion({repr(self.origin)}, {repr(self.relation.relation)}, {repr(self.relation.ref_val)}, {repr(self.ftr_idx)}, {repr(self.cls_idx)})"


class Action(object):
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

    def __init__(self, func, index):
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
        # Only check if this specific feature is binary
        # so that this code will work on mixed representations as well
        assert_binary(sample.f_vals[self.index])
        # can add substructure only if it is not there yet...
        return sample.f_vals[self.index] != self.new_value

    def __str__(self, ):
        return f"f_val[{self.index}] -> {self.new_value}"

    def __repr__(self, ):
        return f"Action({repr(self.desc)}, {repr(self.index)})"

    def equals(self, other):
        # define own comparator method but don't overwrite __eq__
        return (self.index == other.index) and (self.new_value == other.new_value)
