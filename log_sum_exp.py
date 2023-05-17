"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Modified _column_grad by Takayama in 2023-01 for STLCCP

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy 


class log_sum_exp(AxisAtom):
    """:math:`\\log\\sum_i e^{x_i}`

    """

    def __init__(self, x, axis=None, keepdims: bool = False) -> None:
        super(log_sum_exp, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, sums, and takes the log.
        """
        return logsumexp(values[0], axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        n = len(value)
        s = 0
        e = np.zeros(n)
        xmax = np.max(value)
        k = np.argmax(value)
        a = xmax
        
        for i in range(n):
            e[i] = np.exp(value[i] - xmax)
            if i != k:
                s += e[i]

        D = e / (1 + s)
        D = D.reshape((D.shape[0], 1))
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False
