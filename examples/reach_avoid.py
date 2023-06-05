#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
from treelib import Tree
from STLCCP.benchmarks import ReachAvoid
from STLCCP.solver import *
import cvxpy
import matplotlib.patches as mpatches
from typing import Tuple
import time
import os

path = "./fig_reach"
os.makedirs(path, exist_ok=True)

# Specification Parameters
goal_bounds = (7,8,8,9)     # (xmin, xmax, ymin, ymax)
obstacle_bounds = (3,5,4,6)

for i in range(1):

    T = i * 25 + 50

    # Define the specification and system dynamics
    scenario = ReachAvoid(goal_bounds, obstacle_bounds, T)
    sys = scenario.GetSystem()
    spec = scenario.GetSpecification()
    spec.transform2tree().to_graphviz(path+"/notsimplify_reachavoid")
    spec.simplify()
    spec.transform2tree().to_graphviz(path+"/simplify_reachavoid")



    # Specify any additional running cost
    quad = 1e-3
    Q = quad*np.diag([0,0,1,1])   # just penalize high velocities
    R = quad*np.eye(2)

    # Initial state
    x0 = np.array([1.0,2.0,0.0,0])


    # Set bounds on state and control variables
    u_min = np.array([-0.5,-0.5])
    u_max = np.array([0.5, 0.5])
    x_min = np.array([0.0, 0.0, -1.0, -1.0])
    x_max = np.array([10.0, 10.0, 1.0, 1.0])

    # the parameter below determines how many samples we collect with random initial values on variables.
    ccp_times = 1
    for j in range(ccp_times): 
        initial_guess = None
        status = None 
        solve_alltime = 0.0

        # the parameter below determines determines which of lse, mellowmin with lse warm startn, etc. to select 
        # please see Section 7-F of our journal paper. 
        warmstart = 0
        # Set "warmstart = 0" if you want to just solve "lse" smoothed program.
        # Set "warmstart = 1" if you want to use "mellowmin smoothed program with a warm-start solution of lse" smoothed program (lse->mellwmin)".
        # "warmstart >= 2" means (lse->mellwmin->lse->...). 
        for l in range(warmstart+1):
            if l % 2 == 0:
                mode = "lse"
                tempk=10.0
            else:
                mode = "mellowmin"
                tempk=1000.0
            
            solver = CCPSTLSolver(spec, sys, x0, T, k=tempk, mode=mode, verbose=True) 

            # Add bounds
            solver.AddControlBounds(u_min, u_max)
            solver.AddStateBounds(x_min, x_max)

            # Add quadratic running cost (optional)
            solver.AddQuadraticCost(Q,R)

            # Solve the optimization problem
            # weight parameter (default is "tree")
                # "tree": tree weighted penalty
                # "decay": tree weighted penalty with deacay
                # "50", "smallest": penalize equally using the weight 50 or the smallest number of child nodes
            x, u,y, rho_orig,solve_time,initial_guess,status = solver.Solve(initial_guess=initial_guess, weight="tree")
            solve_alltime += solve_time

            # Csv data
            base = os.path.join(path, str(T)+ "quad" + str(quad))
            csvfile = os.path.join(base + ".csv")
            array = []
            array.extend([solve_alltime, rho_orig, T, tempk])
            arr = np.array(array)
            with open(csvfile,"a") as f:
                    np.savetxt(f, [arr], delimiter=",",fmt="%.5f")

        # Plot the solution
        filename = os.path.join(base + ".pdf") 
        if (x is not None) and (rho_orig>0):
            for t in range(T+1):
                plt.scatter(x[0,t].value,x[1,t].value, c="b")
                
                if t!=T:
                    left = np.array([x[0,t].value,x[0,t+1].value])
                    height = np.array([x[1,t].value,x[1,t+1].value])
                    plt.plot(left, height, c="royalblue")
        plt.savefig(filename)


    ax = plt.gca()
    scenario.add_to_plot(ax)
    plt.savefig(filename)

plt.show()