import pymanopt
import numpy as np
import argparse
import logging

from cryomap_align.gauss_proc_utils import calc_corr, calc_grad_corr


def run_manifold_opt(
    surr_coeff: np.ndarray, rot_cand: np.ndarray, config: argparse.Namespace
):
    """
    Run manifold optimization to find new candidate rotation matrices

    Parameters
    ----------
    surr_coeff : np.ndarray
        Surrogate coefficients
    rot_cand : np.ndarray
        Rotation matrices
    config : argparse.Namespace
        Configuration object containing max_iter, min_grad, min_step, verbosity

    Returns
    -------
    np.ndarray
        New candidate rotation matrix
    """
    if config.invert_handedness:
        manifold = pymanopt.manifolds.Stiefel(3, 3)

    else:
        manifold = pymanopt.manifolds.SpecialOrthogonalGroup(3)

    @pymanopt.function.numpy(manifold)
    def cost(X):
        k_x = np.array(
            [calc_corr(X, rot_cand[j], config) for j in range(rot_cand.shape[0])]
        )
        return np.dot(k_x, surr_coeff)

    @pymanopt.function.numpy(manifold)
    def grad(X):
        kx_grad = np.array(
            [calc_grad_corr(X, rot_cand[j], config) for j in range(rot_cand.shape[0])]
        )

        return np.einsum("ijk, i -> jk", kx_grad, surr_coeff)

    problem = pymanopt.Problem(manifold=manifold, cost=cost, euclidean_gradient=grad)

    optimizer = pymanopt.optimizers.SteepestDescent(
        max_iterations=config.manopt_max_iter,
        min_gradient_norm=config.manopt_min_grad,
        min_step_size=config.manopt_min_step,
        verbosity=config.manopt_verbosity,
    )

    result = optimizer.run(problem)

    new_cand = result.point.astype(np.float32)

    return new_cand
