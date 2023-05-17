from .base import STLSolver
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple
from .dccp import *
from ..STL import *
import time

class CCPSTLSolver(STLSolver):
    def __init__(self, spec, sys, x0, T, k, mode, verbose):
        STLSolver.__init__(self, spec, sys, x0, T, verbose)
        self.k = k
        self.mode = mode
        self.Q = np.zeros((sys.n,sys.n))
        self.R = np.zeros((sys.m,sys.m))

        self.y = cvxpy.Variable((self.sys.p, self.T+1)) # y_0,...,y_T-1
        self.x = cvxpy.Variable((self.sys.n, self.T+1)) # x_0,...,x_T-1
        self.u = cvxpy.Variable((self.sys.m, self.T+1)) 

        self.constr = []
        self.constr_penalty = []
        self.cost = 0
        self.AddDynamicsConstraints()

        self.AddRobustness() # Robustness Decomposition (include cost function and constraints)

        # Below are specified in "example" file
            # self.AddControlBounds()
            # self.AddStateBounds()
            # self.AddQuadraticCost(Q,R)


    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T+1):
            self.constr += [self.u[:,t]- u_max <= 0]
            #subformula.countleaves()
            self.constr_penalty += [1]

            self.constr += [ u_min - self.u[:,t] <= 0]
            #subformula.countleaves()
            self.constr_penalty += [1]

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T+1):
            self.constr += [self.x[:,t] - x_max <= 0]
            self.constr_penalty += [1]

            self.constr += [x_min - self.x[:,t] <= 0 ] 
            self.constr_penalty += [1]


    def AddQuadraticCost(self, Q, R):
        for t in range(self.T+1):
            temp = cvxpy.quad_form( self.x[:,t],Q ) + cvxpy.quad_form(self.u[:,t], R )
            if temp.curvature ==  "UNKNOWN":
                print("Quadratic objective curvature unknown")
            else:
                #print(temp.curvature)
                self.cost += temp  


    def AddDynamicsConstraints(self):
        self.constr += [  self.x[:,0] - self.x0 == 0]
        self.constr_penalty += [1]

        for t in range(self.T):
            self.constr += [self.x[:,t+1] - self.sys.f(self.x[:,t], self.u[:,t]) == 0]
            self.constr_penalty += [1]

            self.constr += [self.y[:,t] - self.sys.g(self.x[:,t], self.u[:,t]) == 0]
            self.constr_penalty += [1]

        self.constr += [self.y[:,self.T] - self.sys.g(self.x[:,self.T], self.u[:,self.T]) == 0]
        self.constr_penalty += [1]


    def Solve(self,initial_guess,weight):

        prob = cvxpy.Problem(cvxpy.Minimize(self.cost), self.constr)

        if initial_guess is not None:
            for i,var in enumerate(prob.variables()):
                var.value = initial_guess[i]

        # GUROBI QP solver options:
        #   solopts={'Method':2,'BarConvTol':1e-3,'BarQCPConvTol':1e-3} # barrier
        #   solopts={'Method':1} # 1=dual simplex
        result,solve_time = prob.solve(method="dccp", solver='GUROBI',verbose=False, constr_penalty=self.constr_penalty,weight=weight) # ,verbose=True, **solopts)

        # Output
        print("cost value =", "{:.4f}".format(result[0]),"self.rho_max", "{:.4f}".format(self.rho_max.value[0]))
        print("mode:", self.mode, ", smooth parameter k:", self.k)
        rho_orig = self.spec.orig_robustness(self.y.value,0)[0]
        print("original robustness:", "{:.4f}".format(rho_orig))
        rho = self.spec.robustness(self.y.value,0,self.k,self.mode)
        print("smooth reversed-robustness:", "{:.4f}".format(rho))
        print("solve_time:","{:.2f}".format(solve_time))

        return (self.x,self.u,self.y,rho_orig,solve_time,result[-2],result[-1])


    def AddRobustness(self):
        if self.spec.combination_type =="and":
            self.rho_max = cvxpy.Variable(1) 
            self.constr += [ self.rho_max <= 0] # this is a dcp constraint
            self.spec.countleaves()
            self.constr_penalty += [self.spec.count] 

            #self.constr += [ -20.0 - self.rho_max <= 0] # # this is a dcp constraint and can be dropped off
            #self.constr_penalty += [1]

            self.cost += self.rho_max 
            for i, subformula in enumerate(self.spec.subformula_list):
                t = self.spec.timesteps[i] 
                if isinstance(subformula, LinearPredicate):
                    y = self.y[:,t]
                    self.constr += [ subformula.b - subformula.a.T@y - self.rho_max <= 0]
                    self.constr_penalty += [1]

                elif isinstance(subformula, NonlinearPredicate):
                    y = self.y[:,t]
                    self.constr += [ -subformula.g(y) - self.rho_max <= 0]
                    self.constr_penalty += [1]
                elif subformula.combination_type =="and":
                        print("You did not simplify the robustness tree")
                else: 
                    self.or_and_loop_to_constr(subformula,self.rho_max,t)

        else: #self.spec.combination_type =="or":
            subrho_array = []
            for i, subformula in enumerate(self.spec.subformula_list):
                t_sub = formula.timesteps[i] 
                if isinstance(subformula, LinearPredicate):
                    y = self.y[:,t+t_sub]
                    subrho = subformula.b - subformula.a.T@y
                    subrho_array.append(subrho)

                elif isinstance(subformula, NonlinearPredicate):
                    y = self.y[:,t+t_sub]
                    subrho = -subformula.g(y)
                    subrho_array.append(subrho)

                elif subformula.combination_type == "or":
                    print("error, _or_ should not be here")

                else: #subformula.combination_type == "and": 
                    subrho = cvxpy.Variable(1) 
                    subrho_array.append(subrho)


                    for l, subsubformula in enumerate(subformula.subformula_list):
                        t_subsub = subformula.timesteps[l]
                        if isinstance(subsubformula, LinearPredicate):
                            y = self.y[:,t+t_sub+t_subsub]
                            self.constr += [ subsubformula.b - subsubformula.a.T@y - subrho <= 0]
                            self.constr_penalty += [1]

                        elif isinstance(subsubformula, NonlinearPredicate):
                            # rho = g(y)
                            y = self.y[:,t+t_sub+t_subsub]
                            self.constr += [ -subusbusbformula.g(y) - subrho <= 0 ]
                            self.constr_penalty += [1]

                        else: 
                            self.or_and_loop_to_constr(subsubformula,subrho,t+t_sub+t_subsub)


            length = len(subrho_array)
            for i in range(length):
                subrho_array[i] = -self.k * subrho_array[i]
            subrho_array = cvxpy.vstack(subrho_array)

            if self.mode =="mellowmin": 
                    smoothedmin = - (cvxpy.log_sum_exp(subrho_array)-np.log(length)) / self.k
            else: # self.mode =="lse": 
                    smoothedmin = - (cvxpy.log_sum_exp(subrho_array)) / self.k

            self.cost += smoothedmin 
            self.constr += [smoothedmin <= 0] 
            self.spec.countleaves()
            self.constr_penalty += [self.spec.count]


    def or_and_loop_to_constr(self,formula,rhomax,t):
        subrho_array = []
        for j , subformula in enumerate(formula.subformula_list):
            t_sub = formula.timesteps[j] 
            if isinstance(subformula, LinearPredicate):
                y = self.y[:,t+t_sub]
                subrho = subformula.b - subformula.a.T@y
                subrho_array.append(subrho)

            elif isinstance(subformula, NonlinearPredicate):
                # rho = g(y)
                y = self.y[:,t+t_sub]
                subrho = -subformula.g(y)
                subrho_array.append(subrho)

            elif subformula.combination_type == "or":
                print("error, _or_ should not be here")

            else: #subformula.combination_type == "and": 
                subrho = cvxpy.Variable(1) 
                subrho_array.append(subrho)


                for l, subsubformula in enumerate(subformula.subformula_list):
                    t_subsub = subformula.timesteps[l]
                    if isinstance(subsubformula, LinearPredicate):
                        y = self.y[:,t+t_sub+t_subsub]
                        self.constr += [ subsubformula.b - subsubformula.a.T@y - subrho <= 0]
                        self.constr_penalty += [1]

                    elif isinstance(subsubformula, NonlinearPredicate):
                        # rho = g(y)
                        y = self.y[:,t+t_sub+t_subsub]
                        self.constr += [ -subusbusbformula.g(y) - subrho <= 0 ]
                        self.constr_penalty += [1]

                    else: 
                        self.or_and_loop_to_constr(subsubformula,subrho,t+t_sub+t_subsub)


        length = len(subrho_array)
        for i in range(length):
            subrho_array[i] = -self.k * subrho_array[i]
        subrho_array = cvxpy.vstack(subrho_array)

        if self.mode =="mellowmin": 
                smoothedmin = - (cvxpy.log_sum_exp(subrho_array)-np.log(length)) / self.k
        else: # self.mode =="lse": 
                smoothedmin = - (cvxpy.log_sum_exp(subrho_array)) / self.k

        self.constr += [smoothedmin - rhomax <= 0] 
        formula.countleaves()
        self.constr_penalty += [formula.count]
