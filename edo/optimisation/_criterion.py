raise DeprecationWarning("This isn't the implementation you're looking for.")

class Criterion(object):
    """
    Criterion which Sample's SHAP values must satisfy for Rule to be applied
    
    Criterion has one of the following forms:
    - shap_value relation reference_value, ex: shap_value < 1;
    - left_ref_val left_rel shap_value right_rel right_ref_val, ex: 2 < shap_value <= 5
    
    origin: Origin
        inf. about the model for which Criterion is defined
        used to check if Sample's SHAP values are derived for the same model
    relation: function (float,float) -> boolean (or iterable of such functions)
        required relation of the Sample's SHAP value to the reference_value,
        preferably from the operator module, ex. operator.lt
    reference_value: float (or iterable of floats)
        value to which Sample's SHAP value is compared
    feature_index: int
        index of feature for which Criterion is defined
    class_index: int or None
        in the case of classification index of class for which Criterion is defined,
        in the case of regression should be None
    """

    
import operator
from collections.abc import Iterable
from abc import ABC, abstractmethod
import numpy as np

from .. import make_origin


# To jest makabrycznie rozbuchana abstrakcja, pozwalająca na łatwe sprawdzenie, czy dwa przedziały mają puste przecięcie


def make_criterion(origin, relation, reference_value, feature_index, class_index):
    if relation in CriterionOp._allowed:
        return CriterionOp(origin, relation, reference_value, feature_index, class_index)
    elif isinstance(relation, Iterable) and np.all(r in CriterionOp._allowed for r in relation):
        return CriterionOpDouble(origin, relation, reference_value, feature_index, class_index)
    else:
        return Criterion(origin, relation, reference_value, feature_index, class_index)
    

class Criterion(ABC):
    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        self.origin = make_origin(origin)
        self.relation = relation
        self.ref_val = reference_value
        self.ftr_idx = feature_index
        self.cls_idx = class_index       
    
    @abstractmethod
    def relation_satisfied(self, x):
        # typically self.relation(x, self.ref_val)
        # if there are more relations then np.all(r(x, self.ref_val) for r in self.relation)
        pass
    
    @abstractmethod
    def to_interval(self, ):
        # should return left_value, left_inclusive, right_value, right_inclusive
        pass
    
    @property
    def _x(self, ):
        cls_idx = f"[{self.cls_idx}]" if self.cls_idx is not None else ''
        return f"shap_val{cls_idx}[{self.ftr_idx}]"
    
    def is_satisfied(self, sample):
        assert self.origin == sample.s_vals_origin, f"Cannot check criterion for SHAP values calculated for a different origin! Criterion origin is {self.origin} != {sample.s_vals_origin} (SHAP values origin)."
        
        if self.cls_idx is None:
            s_vals = sample.s_vals[self.ftr_idx]
        else:
            s_vals = sample.s_vals[self.cls_idx, self.ftr_idx]
            
        return self.relation_satisfied(s_vals)
        
    def overlaps(self, other):
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
        
        if c>b:
            # a-----b
            #          c-----d
            return False
        elif c<b:
            # a-----b
            #    c-----d
            return True
        else:
            # a-----b
            #       c-----d
            # if both are inclusive then there is overlap
            return (b_incl and c_incl)
        

# CriterionOp nie defuniuje żadnego API, więc może to mogłaby być funkcja z assertami i kilka słowników luzem?        
class CriterionOp(ABC, Criterion):
    # supports operator.lt, operator.gt, operator.le, operator.ge
    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        if isinstance(relation, Iterable):
            assert np.all(r in CriterionOp._allowed for r in relation), f"{relation} is not in {CriterionOp._allowed}."
        else:
            assert relation in CriterionOp._allowed, f"{relation} is not in {CriterionOp._allowed}."
            
        super().__init__(origin, relation, reference_value, feature_index, class_index)
    
    _allowed = [operator.lt, operator.gt, operator.le, operator.ge]
    _mutually_inclusive = [operator.le, operator.ge]
    _pointing_left = [operator.le, operator.lt]
    _relnames = {'lt':'<', 'le': '≤', 'gt':'>', 'ge':'≥'}  # for pretty printing    
    

class CriterionOpSingle(CriterionOp):
    # x relation reference_value, ex: x < 1;
    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        super().__init__(origin, relation, reference_value, feature_index, class_index)
    
    def relation_satisfied(self, x):
        return self.relation(x, self.ref_val)
    
    def to_interval(self, ):
        if self.relation in CriterionOp._pointing_left:
            l_val = -np.inf
            l_inclusive = False
            r_val = self.ref_val
            r_inclusive = self.relation in CriterionOp._mutually_inclusive
        else:
            l_val = self.ref_val
            l_inclusive = self.relation in CriterionOp._mutually_inclusive
            r_val = np.inf
            r_inclusive = False
            
        return l_val, l_inclusive, r_val, r_inclusive
    
    def __str__(self, ):
        return f"{self._x} {CriterionOp._relnames[self.relation.__name__]} {self.ref_val}"
        
        
class CriterionOpDouble(CriterionOp):
    # left_reference_value left_relation x right_relation right_reference_value, ex. 2 < x <= 3
    def __init__(self, origin, relation, reference_value, feature_index, class_index):
        assert len(relation) == 2, f"Exactly two relations are required, given {len(relation)}."
        assert len(reference_value) == 2, f"Exactly two reference values are required, given {len(reference_value)}."
        assert reference_value[0] <= reference_value[1], f"Required that reference_value[0] <= reference_value[1], given {reference_value}."  # so that we always have 2 < x < 4 and never 2 < x or x > 5
        assert (relation[0] in CriterionOp._pointing_left) != (relation[1] in CriterionOp._pointing_left), f"Relations must point in the same direction, given {relation}."
        
        super().__init__(origin, relation, reference_value, feature_index, class_index)

    # properties are better than alias in __init__ bc changing self.ref_val will 'update' l/r_ref_val as well
    @property
    def l_ref_val(self, ):
        return self.ref_val[0]
    
    @property
    def r_ref_val(self, ):
        return self.ref_val[1]
    
    @property
    def l_relation(self, ):
        return self.relation[0]
    
    @property
    def r_relation(self, ):
        return self.relation[1]
    
    def relation_satisfied(self, x):
        return self.l_relation(x, self.l_ref_val) and self.r_relation(x, self.r_ref_val)
    
    def to_interval(self, ):
        l_inclusive = self.l_relation in CriterionOp._mutually_inclusive
        r_inclusive = self.r_relation in CriterionOp._mutually_inclusive
        return self.l_ref_val, l_inclusive, self.r_ref_val, r_inclusive
    
    def __str__(self, ):
        return f"{self.l_ref_val} {CriterionOp._relnames[self.l_relation.__name__]} {self._x} {CriterionOp._relnames[self.r_relation.__name__]} {self.r_ref_val}"
    