__author__ = "Xinyue"
# Modified by Takayama in 2023-01 for STLCCP

import numpy as np
import cvxpy as cvx


def linearize(expr):
    """Returns the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

    Args:
        expr: An expression.

    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    else:
        if expr.value is None:
            raise ValueError(
                "Cannot linearize non-affine expression with missing variable values."
            )
        tangent = np.real(expr.value) #+ np.imag(expr.value)
        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            complex_flag = False
            if var.is_complex() or np.any(np.iscomplex(grad_map[var])):
                complex_flag = True
            if var.ndim > 1:
                temp = cvx.reshape(
                    cvx.vec(var - var.value), (var.shape[0] * var.shape[1], 1)
                )
                if complex_flag:
                    flattened = np.transpose(np.real(grad_map[var])) @ cvx.real(temp) + \
                    np.transpose(np.imag(grad_map[var])) @ cvx.imag(temp)
                else:
                    flattened = np.transpose(np.real(grad_map[var])) @ temp
                tangent = tangent + cvx.reshape(flattened, expr.shape)
            elif var.size > 1:
                if complex_flag:
                    tangent = tangent + np.transpose(np.real(grad_map[var])) @ (cvx.real(var) - np.real(var.value)) \
                    + np.transpose(np.imag(grad_map[var])) @ (cvx.imag(var) - np.imag(var.value))
                else:
                    tangent = tangent + np.transpose(np.real(grad_map[var])) @ (var - var.value)
            else:
                if complex_flag:
                    tangent = tangent + np.real(grad_map[var]) * (cvx.real(var) - np.real(var.value)) \
                    + np.imag(grad_map[var]) * (cvx.imag(var) - np.imag(var.value))
                else:
                    tangent = tangent + np.real(grad_map[var]) * (var - var.value)
        return tangent
