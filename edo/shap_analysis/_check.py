from .. import Task, TASK_ERROR_MSG

        
def validate_task(task, s_vals):
    # TODO: dwa razy sprawdzamy to samo, choose one
    assert (task == Task.CLASSIFICATION and len(s_vals.shape) == 2) or (task == Task.REGRESSION and len(s_vals.shape) == 1), f"`s_vals.shape` and task mismatch. `s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape} and task {task}."
    assert (len(s_vals.shape), task) in [(1, Task.REGRESSION), (2, Task.CLASSIFICATION)], f"`s_vals.shape` and `task` mismatch. `s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape} and task is {task}."



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
