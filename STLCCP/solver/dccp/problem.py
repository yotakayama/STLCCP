__author__ = "Xinyue"
# Modified by Takayama in 2023-01 for STLCCP

import numpy as np
import cvxpy as cvx
import logging
import warnings
import time
from .objective import convexify_obj
from .constraint import convexify_constr
import os

logger = logging.getLogger("dccp")
logger.addHandler(logging.FileHandler(filename="dccp.log", mode="w", delay=True))
logger.setLevel(logging.INFO)
logger.propagate = False


def dccp(
    self,
    max_iter=25,
    tau=0.005,
    mu=2.0,
    tau_max=1e3,
    solver=None,
    max_penalty=1e-5,
    ep=1e-2,
    **kwargs
):
    """
    main algorithm ccp
    :param max_iter: maximum number of iterations in ccp
    :param tau: initial weight on slack variables
    :param mu:  increment of weight on slack variables
    :param tau_max: maximum weight on slack variables
    :param solver: specify the solver for the transformed problem
    :return
        if the transformed problem is infeasible, return None;
    """
    self.tau = tau
    print("Solver:",solver)
    if not is_dccp(self):
        raise Exception("Problem is not DCCP.")
    if is_contain_complex_numbers(self):
        warnings.warn("Problem contains complex numbers and may not be supported by DCCP.")
        logger.info("WARN: Problem contains complex numbers and may not be supported by DCCP.")
    result = None
    if self.objective.NAME == "minimize":
        cost_value = float("inf")  # record on the best cost value
    else:
        cost_value = -float("inf")
    start_time=time.time()
    ccp_times = 1 
    # In STLCCP, this ccp_times parameter, which determines how many times we run ccp to solve a problem
    #  with random initial values on variables, is 
    for t in range(ccp_times):  # for each time of running ccp
        dccp_ini(
            self, random=(ccp_times > 1), solver=solver, **kwargs
        )  # initialization; 
        # iterations
        result_temp = iter_dccp(
            self, max_iter, self.tau, mu, tau_max, solver, ep, max_penalty, **kwargs
        )
        # first iteration
        if t == 0:
            self._status = result_temp[-1]
            result = result_temp
            cost_value = result_temp[0]
            result_record = {}
            for var in self.variables():
                result_record[var] = var.value
        else:
            if result_temp[-1] == "Converged":
                self._status = result_temp[-1]
                if result_temp[0] is not None:
                    if (
                        (cost_value is None)
                        or (
                            self.objective.NAME == "minimize"
                            and result_temp[0] < cost_value
                        )
                        or (
                            self.objective.NAME == "maximize"
                            and result_temp[0] > cost_value
                        )
                    ):  # find a better cost value
                        # no slack; slack small enough
                        if len(result_temp) < 4 or result_temp[1] < max_penalty:
                            result = result_temp  # update the result
                            cost_value = result_temp[
                                0
                            ]  # update the record on the best cost value
                            for var in self.variables():
                                result_record[var] = var.value
            else:
                for var in self.variables():
                    var.value = result_record[var]
    solve_time = time.time() - start_time
    return result,solve_time


def dccp_ini(self, times=1, random=0, solver=None, **kwargs):
    """
    set initial values
    :param times: number of random projections for each variable
    :param random: mandatory random initial values
    """
    dom_constr = self.objective.args[0].domain  # domain of the objective function
    for arg in self.constraints:
        for l in range(len(arg.args)):
            for dom in arg.args[l].domain:
                dom_constr.append(dom)  # domain on each side of constraints
    var_store = {}  # store initial values for each variable
    init_flag = {}  # indicate if any variable is initialized by the user
    var_user_ini = {}
    #print("self.variables()",self.variables())
    for var in self.variables():
        var_store[var] = np.zeros(var.shape)  # to be averaged
        init_flag[var] = var.value is None
        if var.value is None:
            var_user_ini[var] = np.zeros(var.shape)
        else:
            var_user_ini[var] = var.value
    for t in range(times):  # for each time of random projection
        # setup the problem
        ini_cost = 0
        for var in self.variables():
            if (init_flag[var] or random):  # if the variable is not initialized by the user, or random initialization is mandatory
                if len(var.shape) > 1:
                    a = cvx.norm(var - np.random.randn(var.shape[0], var.shape[1]) * 10, "fro")
                    ini_cost += a
                else:
                    a = cvx.norm(var - np.random.randn(var.size) * 10)
                    ini_cost += a
        ini_obj = cvx.Minimize(ini_cost)
        ini_prob = cvx.Problem(ini_obj, dom_constr)

        solver = "ECOS" # only for the above problem to avoid error
        #print(dom_constr)
        ini_prob.solve(solver=solver)  
        print("End solving initial problem")

        for var in self.variables():
            var_store[var] = var_store[var] + var.value / float(times)  # randomized 
    # set initial values
    for var in self.variables():
        if init_flag[var] or random: 
            #print("randomized")
            var.value = var_store[var]
        else:
            #print("User specified")
            var.value = var_user_ini[var]


def is_dccp(problem):
    """
    :param
        a problem
    :return
        a boolean indicating if the problem is dccp
    """
    if problem.objective.expr.curvature == "UNKNOWN":
        return False
    for constr in problem.constraints:
        for arg in constr.args:
            if arg.curvature == "UNKNOWN":
                return False
    return True

def is_contain_complex_numbers(self):
    for variable in self.variables():
        if variable.is_complex():
            return True
    for para in self.parameters():
        if para.is_complex():
            return True
    for constant in self.constants():
        if constant.is_complex():
            return True
    for arg in self.objective.args:
        if arg.is_complex():
            return True
    for constr in self.constraints:
        for arg in constr.args:
            if arg.is_complex():
                return True
    return False


def iter_dccp(self, max_iter, tau, mu, tau_max, solver, ep, max_penalty_tol, **kwargs):
    """
    ccp iterations
    :param max_iter: maximum number of iterations in ccp
    :param tau: initial weight on slack variables
    :param mu:  increment of weight on slack variables
    :param tau_max: maximum weight on slack variables
    :param solver: specify the solver for the transformed problem
    :return
        value of the objective function, maximum value of slack variables, value of variables
    """
    # split non-affine equality constraints
    constr = []
    for constraint in self.constraints:
        if (str(type(constraint)) == "<class 'cvxpy.constraints.zero.Equality'>" and not constraint.is_dcp()):
            print("The optimization problem contains some non-affine equality constraints that CCP has to approximate, which can decrease the performance of our method.")
            print("Note that the robustness decomposition method does not produce any equality constraints")
            constr.append(constraint.args[0] <= constraint.args[1])
            constr.append(constraint.args[0] >= constraint.args[1])
        else:
            constr.append(constraint)
    obj = self.objective
    tau = self.tau
    self = cvx.Problem(obj, constr)
    self.tau = tau


    it = 1
    converge = False

    # keep the values from the previous iteration or initialization
    previous_cost = float("inf")
    previous_org_cost = self.objective.value
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)

    # each non-dcp constraint needs a slack variable
    var_penalty = []
    var_weight =[]


    self.constr_penalty = kwargs['constr_penalty']
    kwargs.pop('constr_penalty')
    weight_type = kwargs['weight']
    kwargs.pop('weight')
    
    for i, constr in enumerate(self.constraints):
        if (not constr.is_dcp()):
            var_penalty.append(cvx.Variable(constr.shape))
            var_weight.append(self.constr_penalty[i]) # tree weighted penalty (TWP)

    # Just compute the average and minimum of penalty weights
    sum=0
    for i in range(len(var_weight)):
        sum += var_weight[i]
    av = float(sum)/ len(var_weight)
    penaltymin=np.array(var_weight).min() 
    if weight_type == "50":
        var_weight = np.full_like(var_weight,50) # same-weight = 50 
    elif weight_type == "smallest":
        var_weight = np.full_like(var_weight,penaltymin) # same-weight = lowest in tree

    rate=0.2 # the decay parameter for TWP-CCP with decay

    while it <= max_iter and all(var.value is not None for var in self.variables()):

        constr_new = []
        # objective
        convexified_obj = convexify_obj(self.objective)
        if not self.objective.is_dcp():
            # non-sub/super-diff
            while convexified_obj is None:
                # damping
                var_index = 0
                for var in self.variables():
                    var.value = 0.8 * var.value + 0.2 * variable_pres_value[var_index]
                    var_index += 1
                convexified_obj = convexify_obj(self.objective)
            # domain constraints
            for dom in self.objective.expr.domain:
                print(dom,"dom")
                constr_new.append(dom)
        # new cost function
        cost_new = convexified_obj.expr

        # constraints
        count_slack = 0
        for i,arg in enumerate(self.constraints):
            temp = convexify_constr(arg) 
            if (not arg.is_dcp()): 
                while temp is None:
                    # damping
                    var_index = 0
                    for var in self.variables():
                        var.value = (
                            0.8 * var.value + 0.2 * variable_pres_value[var_index]
                        )
                        var_index += 1
                    temp = convexify_constr(arg)

                newcon = temp[0]  
                for dom in temp[1]:  
                    constr_new.append(dom)
                constr_new.append(newcon.expr <= var_penalty[count_slack])
                constr_new.append(var_penalty[count_slack] >= 0)
                count_slack = count_slack + 1
                
            else:
                constr_new.append(arg)

        # objective
        if self.objective.NAME == "minimize":
            for i,var in enumerate(var_penalty):
                if weight_type == "decay":
                    cost_new += ((var_weight[i]-penaltymin) * np.exp(-rate*(it-1)) + penaltymin) * self.tau * cvx.sum(var)
                else:
                    cost_new += var_weight[i]  * self.tau * cvx.sum(var)
            obj_new = cvx.Minimize(cost_new)
        else:
            for i,var in enumerate(var_penalty):
                if weight_type == "decay":
                    cost_new -= ((var_weight[i]-penaltymin) * np.exp(-rate*(it-1)) + penaltymin) * self.tau * cvx.sum(var)
                else:
                    cost_new -= var_weight[i]  * self.tau * cvx.sum(var)
            obj_new = cvx.Maximize(cost_new)

        # new problem
        #print(len(constr_new),"length of constr new")
        prob_new = cvx.Problem(obj_new, constr_new)

        # keep previous value of variables
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        # solve
        if solver is None:
            prob_new_cost_value = prob_new.solve(**kwargs)
        else:
            prob_new_cost_value = prob_new.solve(solver=solver, **kwargs)
        if prob_new_cost_value is not None:
            logger.info(
                "iteration=%d, cost value=%.5f, self.tau=%.5f, solver status=%s",
                it,
                prob_new_cost_value,
                self.tau,
                prob_new.status,
            )
        else:
            logger.info(
                "iteration=%d, cost value=%.5f, self.tau=%.5f, solver status=%s",
                it,
                np.nan,
                self.tau,
                prob_new.status,
            )
        

        max_penalty = None
        # print slack
        if (
            prob_new._status == "optimal" or prob_new._status == "optimal_inaccurate"
        ) and not var_penalty == []:
            slack_values = [v.value for v in var_penalty if v.value is not None]
            max_penalty = max([np.max(v) for v in slack_values] + [-np.inf])
            logger.info("max slack = %.5f", max_penalty)
        # terminal conditions
        if (
            (prob_new.value is not None
            and np.abs(previous_cost - prob_new.value) <= ep
            and np.abs(self.objective.value - previous_org_cost) <= ep
            and (max_penalty is None or max_penalty <= max_penalty_tol))
        ):
            it = max_iter + 1
            converge = True
        else:
            print("Iteration: ",it, ", max_penalty:", "{:.2e}".format(max_penalty), " tau: " , self.tau)
            print("Terminal Conditions:" #, (prob_new.value is not None) # relaxed program has a value
            , (np.abs(previous_cost - prob_new.value) <= ep) #  cost_difference (relaxed program)
            , (np.abs(self.objective.value - previous_org_cost) <= ep) # cost_difference (original program)
            , (max_penalty is None or max_penalty <= max_penalty_tol)   ) #  max_penalty  
            print("--- Constraints whose penalty variables remain high ---")
            for i in range(len(slack_values)):
                if slack_values[i] >= 1.0: #or (var_weight[i] != 4):
                    print("constraint number:", i, ", slack value:" ,"{:.4f}".format(slack_values[i][0]),", penalty weight:", var_weight[i]) 
            print("--------------------------------------")

            previous_cost = prob_new.value
            previous_org_cost = self.objective.value
            self.tau = min([self.tau * mu, tau_max])
            it += 1


    # return
    if converge:
        self._status = "Converged"
    else:
        self._status = "Not_converged"
    var_value = []
    for var in self.variables():
        var_value.append(var.value)
    if not var_penalty == []:
        return (self.objective.value, max_penalty, var_value, self._status)
    else:
        return (self.objective.value, var_value, self._status)


cvx.Problem.register_solve("dccp", dccp)
