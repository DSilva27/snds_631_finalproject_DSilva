import argparse
import numpy as np
import pymanopt
from aspire.volume import Volume
import logging

from cryomap_align.wavelet_utils import vol_to_dwt, calc_loss
from cryomap_align.gauss_proc_utils import calc_corr
from cryomap_align.manopt_utils import run_manifold_opt


def run_gaussian_opt(
    vol_obj: Volume,
    vol_ref: Volume,
    init_cand: np.ndarray,
    config: argparse.Namespace,
):
    assert vol_obj.shape == vol_ref.shape, "Volumes must have the same shape"
    assert vol_obj.ndim == 4, "Volumes must be 3D"  # 4D for batch processing
    assert (
        config.downsample_res < vol_obj.shape[1]
    ), "Downsampling resolution must be smaller than volume size"
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

    vol_obj_ds = vol_obj.downsample(config.downsample_res)
    vol_ref_ds = vol_ref.downsample(config.downsample_res)
    vol_ref_dwt = vol_to_dwt(vol_ref_ds._data[0])

    # allocate memory for new candidates and initialize
    rot_cand = np.zeros((config.max_iter, 3, 3), dtype=np.float32)
    dist_vect = np.zeros(config.max_iter)
    corr_mtx = np.zeros((config.max_iter, config.max_iter))

    for i in range(init_cand.shape[0]):
        rot_cand[i] = init_cand[i]
        dist_vect[i] = calc_loss(rot_cand[i], vol_obj_ds, vol_ref_dwt)

        for j in range(init_cand.shape[0]):
            corr_mtx[i, j] = calc_corr(rot_cand[i], rot_cand[j], config)

    for it in range(init_cand.shape[0], config.max_iter):
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
        dist_vect[it] = calc_loss(new_rot, vol_obj_ds, vol_ref_dwt)
        rot_cand[it] = new_rot

        for i in range(it):
            new_corr = calc_corr(new_rot, rot_cand[i], config)
            corr_mtx[it, i] = new_corr
            corr_mtx[i, it] = new_corr

        corr_mtx[it, it] = 1

    opt_rot = rot_cand[dist_vect.argmin(0)]

    return opt_rot
