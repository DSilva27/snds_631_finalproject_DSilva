import argparse
import numpy as np
import pymanopt
from aspire.volume import Volume
import logging
import warnings
from tqdm import tqdm

from cryomap_align.wavelet_utils import vol_to_dwt, calc_wemd
from cryomap_align.gauss_proc_utils import calc_corr
from cryomap_align.manopt_utils import run_manifold_opt
from cryomap_align.utils import calc_l2_loss


def run_gaussian_opt(
    vol: Volume,
    vol_ref: Volume,
    init_cand: np.ndarray,
    config: argparse.Namespace,
):
    """
    Run the Gaussian process optimization algorithm.

    Parameters
    ----------
    vol : aspire.volume.Volume
        Volume to be aligned.
    vol_ref : aspire.volume.Volume
        Reference volume.
    init_cand : np.ndarray
        Initial candidates for the rotation matrix.
    config : argparse.Namespace
        Configuration object (from config file).

    Returns
    -------
    opt_rot : np.ndarray
        Optimal rotation matrix.

    """
    assert vol.shape == vol_ref.shape, "Volumes must have the same shape"
    assert vol.ndim == 4, "Volumes must be 3D"  # 4D for batch processing
    assert (
        config.max_iter > init_cand.shape[0]
    ), "Max iter must be greater than init_cand"
    assert init_cand.shape[0], "Init cand must have at least one candidate"
    assert init_cand.ndim == 3, "Init cand must be 3D"

    for i in range(init_cand.shape[0]):
        assert np.allclose(
            init_cand[i].T @ init_cand[i], np.eye(3)
        ), "Init cand must be orthogonal"

    assert config.regul_const > 0, "Tau must be greater than 0"
    assert config.regul_const < 1e-1, "Tau must be smaller than 1e-1"

    if config.loss_type == "wemd":
        vol_ref_dwt = vol_to_dwt(vol_ref._data[0])
        warnings.filterwarnings("ignore")

    # allocate memory for new candidates and initialize
    rot_cand = np.zeros((config.max_iter, 3, 3), dtype=np.float32)
    dist_vect = np.zeros(config.max_iter)
    corr_mtx = np.zeros((config.max_iter, config.max_iter))

    for i in range(init_cand.shape[0]):
        rot_cand[i] = init_cand[i]

        if config.loss_type == "wemd":
            dist_vect[i] = calc_wemd(rot_cand[i], vol, vol_ref_dwt)
        else:
            dist_vect[i] = calc_l2_loss(rot_cand[i], vol, vol_ref)

        for j in range(init_cand.shape[0]):
            corr_mtx[i, j] = calc_corr(rot_cand[i], rot_cand[j], config)

    with tqdm(range(init_cand.shape[0], config.max_iter)) as pbar:
        for it in pbar:
            surr_coeff = np.linalg.solve(
                corr_mtx[:it, :it] + config.regul_const * np.eye(it),
                dist_vect[:it],
            )

            # run manifold optimization
            new_rot = run_manifold_opt(
                surr_coeff,
                rot_cand[:it],
                config,
            )

            if config.loss_type == "wemd":
                dist_vect[it] = calc_wemd(new_rot, vol, vol_ref_dwt)

            else:
                dist_vect[it] = calc_l2_loss(new_rot, vol, vol_ref)

            rot_cand[it] = new_rot

            for i in range(it):
                new_corr = calc_corr(new_rot, rot_cand[i], config)
                corr_mtx[it, i] = new_corr
                corr_mtx[i, it] = new_corr

            corr_mtx[it, it] = 1

            pbar.set_postfix(min_loss=np.min(dist_vect[: it + 1]))

    opt_rot = rot_cand[dist_vect.argmin(0)]

    return opt_rot
