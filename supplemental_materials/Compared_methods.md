### Compared Methods
To reproduce the results of two other methods we compared in our [paper](https://arxiv.org/), we refer to [stlpy](https://stlpy.readthedocs.io/en/latest/getting_started.html#a-simple-example) for the detail. The outline for each method is as follows:


#### GUROBI-MIP method
1. Install [stlpy](https://github.com/vincekurtz/stlpy) library.
2. Ensure that you have completed steps 1 and 3 of the [installation procedure](#installation) above.
3. Use the GurobiMICPSolver to solve the specifications of your choice.

#### SNOPT-NLP method
1. Install [stlpy](https://github.com/vincekurtz/stlpy) library.
2. Install the binary version of [Drake](https://drake.mit.edu/) to use [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) with the drake interface. You can refer to either of the following links: [from binary](https://drake.mit.edu/from_binary.html)
and [via pip](https://drake.mit.edu/pip.html#stable-releases).
3. Use the DrakeSmoothSolver among the solvers available in stlpy.