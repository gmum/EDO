from .. import Task, TASK_ERROR_MSG


# ISSUE 131 TODO
def _check_unlogging(unlog, task):
    # regresja - odlogowane shapy
    # klasyfikacja - nieodlogowane shapy
    if task==Task.REGRESSION:
        assert unlog, f"SHAP values were calculated for a regressor that was not unlogged!"
    elif task == Task.CLASSIFICATION:
        assert not unlog, f"SHAP values were calculated for a classifier that was unlogged!"
    else:
        raise ValueError(TASK_ERROR_MSG(task))
