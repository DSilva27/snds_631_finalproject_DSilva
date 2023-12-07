import numpy as np
import argparse
import logging


def calc_corr(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    config: argparse.Namespace,
) -> float:
    """
    Calculate correlation between two points matrices matrix1, matrix2

    Parameters
    ----------
    matrix1 : array
        First matrix of points
    matrix2 : array
        Second matrix of points
    config : argparse.Namespace
        Configuration object containing marg_var and corr_length

    """

    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"
    assert matrix1.ndim == 2, "Matrices must be 2D"

    corr = config.marg_var * np.exp(
        -0.5 * np.linalg.norm(matrix1 - matrix2) ** 2 / config.corr_length**2
    )
    return corr


def calc_grad_corr(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    config: argparse.Namespace,
) -> np.ndarray:
    """
    Calculate gradient of correlation between two points matrices matrix1, matrix2

    Parameters
    ----------
    matrix1 : array
        First matrix of points
    matrix2 : array
        Second matrix of points
    config : argparse.Namespace
        Configuration object containing marg_var and corr_length

    Returns
    -------
    array
        Gradient of correlation between two points matrices matrix1, matrix2
    """

    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"
    assert matrix1.ndim == 2, "Matrices must be 2D"

    grad = (
        calc_corr(matrix1, matrix2, config)
        * (matrix2 - matrix1)
        / config.corr_length**2
    )

    return grad
