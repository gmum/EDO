# from ... import Task, TASK_ERROR_MSG
# from .high_impact import high_impact as _high_impact
# from .unimportant import unimportant as _unimportant
# from .well_separated import two_way_separation, three_way_separation
#
#
# def _calculate(f_vals, s_vals, func, kwargs, task):
#     if task == Task.CLASSIFICATION:
#         r = [func(f_vals, s_vals[c, :], **kwargs) for c in range(s_vals.shape[0])]
#     elif task == Task.REGRESSION:
#         r = func(f_vals, s_vals, **kwargs)
#     else:
#         raise ValueError(TASK_ERROR_MSG(task))
#     return r
#
#
# def well_separated(feature_values, shap_values, task, n_way=2, kwargs={}):
#     # call different function based on n_way
#     assert n_way in [2, 3], f"`n_way` must be 2 or 3, is {n_way}."
#     func = two_way_separation if n_way == 2 else three_way_separation
#     return _calculate(feature_values, shap_values, func, kwargs, task)
#
#
# def high_impact(feature_values, shap_values, task, gamma, metric):
#     func = _high_impact
#     return _calculate(feature_values, shap_values, func, {'gamma':gamma, 'metric':metric}, task)
#
#
# def unimportant(feature_values, shap_values, task, miu, metric):
#     func = _unimportant
#     return _calculate(feature_values, shap_values, func, {'miu':miu, 'metric':metric}, task)
