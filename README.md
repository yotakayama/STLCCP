# STLCCP: An Efficient Convex Optimization-based Framework for Signal Temporal Logic Specifications

STLCCP is an efficient convex-concave procedure (CCP)-based framework that can effectively solve control problems for any STL specifications. STLCCP leverages several structures of STL. The advantage of our method is generally more prominent when the specifications are long-horizon. For more information, please refer to our [publication](#related-publication).


## Requirements
We verified our code with the following version specifications on Apple MBA M1 2020:
- Python 3.10.6
- CVXPY 1.1.19
- Gurobi 10.0

CVXPY Version >= 1.2 may not work. Libraries such as numpy, treelib, matplotlib are also required.

## Installation
1. Install the Gurobi optimizer by following the instructions on the [official website](https://www.gurobi.com/products/gurobi-optimizer/).
If you are eligible, you can obtain a [free license](https://www.gurobi.com/academia/academic-program-and-licenses/).
2. Clone this repository and install.
```
git clone https://github.com/yotakayama/STLCCP
cd STLCCP
pip install .
```
3. Install CVXPY 1.1.19 and replace the file `cvxpy-1.1.19/cvxpy/atoms/log_sum_exp.py` with the file `STLCCP/log_sum_exp.py` in order to avoid numerical issues related to the logsumexp function with high smooth parameter $k$.
```
cd ..
pip install cvxpy==1.1.19
mv STLCCP/log_sum_exp.py cvxpy-1.1.19/cvxpy/atoms/
cd cvxpy-1.1.19
pip install .
```
4. Run the examples.
```
cd ../STLCCP/examples
python3 many_target.py
```
## Parameters
The parameters used in our method are classified into two groups: one that is commonly used in the penalty CCP framework and another specific to our approach. The former set of parameters is listed in the table below.

| Parameters | Description | Default |
|------------|-------------|-------|
| $\tau$ | weight on penalty variables in the cost function at the outset | 5e-3 |
| $\mu$ | rate at which $\tau$ increases | 2.0 |
| $\tau_{\max ,\mu}$ | maximum $\tau$ | 1e3 |
| $s_{\text{terminal}}$ | maximum values on variables for terminal condition | 1e-5 |
| $ep$ | maximum cost difference for terminal condition | 1e-2 |
| solver |  the solver used to solve QP subproblems | GUROBI |
| solopts | the selected solver's option | None | 

<!-- One more parameter is the solver option that chooses what solver to use to solve QP subproblems (note that the options of solvers can be also chosen by `solopts` parameter). The default solver is GUROBI.-->

The values of these commonly used parameters can be modified in the arguments of function `prob.solve()` in `STLCCP/solver/solver.py`.

The latter group of parameters in our method includes the weight parameter, which determines whether the CCP is a _tree-weighted penalty CCP (TWP-CCP)_, a normal penalty CCP, or one of the other weights. The default value is TWP-CCP. The second parameter is the warmstart parameter, which decides whether to use LogSumExp (warmstart=0), Mellowmin (warmstart=1), or other options to smooth the program. We refer to this parameter as the warmstart parameter because, when we use the Mellowmin smoothing method, we use the solution of the LSE-smoothed program as the warmstart (see the experiments in Section 7-F of our [paper](https://arxiv.org/abs/2305.09441) for this approach). The default value is LSE. Both of these parameters can be adjusted in the files in the example folder (e.g., `example/many_target.py`). 


## Related Publication
- Y. Takayama, K. Hashimoto, and T. Ohtsuka, "STLCCP: An Efficient Convex Optimization-based Framework for Signal Temporal Logic Specifications," https://arxiv.org/abs/2305.09441, 2023

```
@article{takayama2023STLCCP,
  title={STLCCP: An Efficient Convex Optimization-based Framework for Signal Temporal Logic Specifications},
  author={Yoshinari Takayama and Kazumune Hashimoto and Toshiyuki Ohtsuka},
  journal={arXiv preprint arXiv:2305.09441},
  year={2023}
}
```

## Some Remarks
### Implementation on CCP part
CVXPY was used as the interface to the solvers, but the CVXPY reduction chain that is performed at each step to normalize the optimization program adds additional computation time which can possibly be removed. For more information, see the cvxpy [tutorial](https://www.cvxpy.org/tutorial/advanced/index.html) and the cvxpy implementation of the convex-concave procedure [DCCP](http://github.com/cvxgrp/dccp/).

### Benchmarks
The benchmarks we used are from a python library for STL called [stlpy](https://github.com/vincekurtz/stlpy). Please see their [documentation](https://stlpy.readthedocs.io/en/latest/getting_started.html#a-simple-example) about the benchmark. 