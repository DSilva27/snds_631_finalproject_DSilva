import configargparse
import os
import logging
import numpy as np
from aspire.volume import Volume
from aspire.utils.rotation import Rotation


def try_mkdir(path):
    if not os.path.exists(path):  # path does not exist, create it
        os.makedirs(path)
        return True

    return True


def init_config(parser: configargparse.ArgumentParser):
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--vol1", type=str, help="Path to volume 1")
    parser.add_argument("--vol2", type=str, help="Path to volume 2")
    parser.add_argument(
        "--downsample_res", type=int, help="Resolution after downsampling", default=None
    )
    parser.add_argument(
        "--pixel_size", type=float, help="Pixel size of volume", default=1.0
    )
    parser.add_argument("--centering_noise_filter", type=float, default=-np.inf)
    parser.add_argument("--center_order", type=int, default=3)
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations",
        default=200,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        help="Loss type to use (l2 or wemd)",
        choices=["l2", "wemd"],
        required=True,
    )
    parser.add_argument(
        "--regul_const",
        type=float,
        help="Regularazing constant for computing the inverse of the corr. matrix K",
        default=1e-3,
    )
    parser.add_argument(
        "--init_cand",
        type=str,
        help="Path to numpy file containing initial candidates",
        default=None,
    )
    parser.add_argument(
        "--marg_var",
        type=float,
        help="Marginal variance for Gaussian Process",
        default=1.0,
    )
    parser.add_argument(
        "--corr_length",
        type=float,
        help="Correlation length for Gaussian Process (default 0.75 if using Wasserstein distance or 1.0 if using l2)",
        default=None,
    )
    parser.add_argument(
        "--manopt_max_iter",
        type=float,
        help="Maximum number of iterations for manifold optimization",
        default=100,
    )

    parser.add_argument(
        "--manopt_min_grad",
        type=float,
        help="Minimum gradient for manifold optimization",
        default=0.1,
    )
    parser.add_argument(
        "--manopt_min_step",
        type=float,
        help="Minimum step for manifold optimization",
        default=0.1,
    )
    parser.add_argument(
        "--manopt_verbosity",
        type=int,
        help="Verbosity for manifold optimization",
        default=0,
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save results", default="."
    )
    parser.add_argument(
        "--invert_handedness",
        type=int,
        help="Volume might have inverse handedness (used in manifold optimization)",
        default=0,
    )

    parser.add_argument(
        "--refine",
        type=bool,
        help="Refine alignment using Nelder-Mead algorithm",
        default=False,
    )

    return


def validate_config(config: configargparse.Namespace):
    # check if the volume files exist
    assert os.path.exists(config.vol1), "Volume 1 does not exist"
    assert os.path.exists(config.vol2), "Volume 2 does not exist"

    # check if both volumes have the same size
    vol1 = Volume.load(config.vol1)
    vol2 = Volume.load(config.vol2)
    assert vol1.shape == vol2.shape, "Volumes must have the same shape"

    # check that pixel size is positive
    assert config.pixel_size > 0, "Pixel size must be positive"

    # check that max iter is positive
    assert config.max_iter > 0, "Max iter must be positive"

    # check that regularization constant is positive
    assert config.regul_const > 0, "Regularization constant must be positive"

    # check that marginal variance is positive
    assert config.marg_var > 0, "Marginal variance must be positive"

    # check that manifold optimization max iter is positive
    assert config.manopt_max_iter > 0, "Manifold optimization max iter must be positive"

    # check that manifold optimization min grad is positive
    assert config.manopt_min_grad > 0, "Manifold optimization min grad must be positive"

    # check that manifold optimization min step is positive
    assert config.manopt_min_step > 0, "Manifold optimization min step must be positive"

    # check that manifold optimization verbosity is positive
    assert (
        config.manopt_verbosity >= 0
    ), "Manifold optimization verbosity must be positive"

    # check downsample resolution
    if config.downsample_res is not None:
        assert (
            config.downsample_res <= vol1.shape[0]
        ), "Downsample resolution must be smaller or equal than volume size"

    # check if the initial candidates file exists
    if config.init_cand is not None:
        assert os.path.exists(
            config.init_cand
        ), "Initial candidates file does not exist"

    if config.corr_length is None:
        if config.loss_type == "wemd":
            config.corr_length = 0.75
        else:
            config.corr_length = 1.0

    return


def print_config(config: configargparse.Namespace):
    logging.info("Experiment name: %s", config.experiment_name)
    logging.info("Volume 1: %s", config.vol1)
    logging.info("Volume 2: %s", config.vol2)
    logging.info("Pixel size: %f", config.pixel_size)
    logging.info("Centering noise filter: %f", config.centering_noise_filter)
    logging.info("Centering order: %d", config.center_order)
    logging.info("Max iter: %d", config.max_iter)
    logging.info("Loss type: %s", config.loss_type)
    logging.info("Regularization constant: %f", config.regul_const)
    logging.info("Initial candidates: %s", config.init_cand)
    logging.info("Marginal variance: %f", config.marg_var)
    logging.info("Correlation length: %f", config.corr_length)
    logging.info("Manifold optimization max iter: %d", config.manopt_max_iter)
    logging.info("Manifold optimization min grad: %f", config.manopt_min_grad)
    logging.info("Manifold optimization min step: %f", config.manopt_min_step)
    logging.info("Manifold optimization verbosity: %d", config.manopt_verbosity)
    logging.info("Save path: %s", config.save_path)
    logging.info("Refine: %s", config.refine)

    return


def calc_l2_loss(rot: np.ndarray, vol_obj: Volume, vol_ref: Volume) -> float:
    """
    Calculate the loss between the objective volume (volume we wish to align) and reference volume. The loss is the Frobenius norm between the objective volume and the reference volume after rotating the objective volume by the rotation matrix.

    Parameters
    ----------
    rot : np.ndarray
        Rotation matrix.
    vol_obj : Volume
        Objective volume.
    vol_ref: Volume
        Reference volume.

    Returns
    -------
    loss : float
        Loss between the objective volume and reference volume.
    """

    vol_obj_rot = vol_obj.rotate(Rotation(rot.T))._data[0]

    loss = np.linalg.norm(vol_obj_rot - vol_ref._data[0])

    return loss
