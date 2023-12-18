import configargparse
from aspire.volume import Volume
from aspire.utils.rotation import Rotation as aspire_Rotation
from scipy.spatial.transform import Rotation as scipy_Rotation
import numpy as np
import os
import logging
import time
import sys

from cryomap_align.utils import init_config, try_mkdir
from cryomap_align.vol_utils import center_vol
from cryomap_align.gauss_opt_utils import run_gaussian_opt
from cryomap_align.opt_refinement import run_nelder_mead_refinement


sys.argv = ["--config", "config.ini"]
parser = configargparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--config",
    is_config_file=True,
    help="Path to config file.",
    required=True,
)

init_config(parser)
config = parser.parse_args(sys.argv)

# create experiment directory
logging.captureWarnings(False)

logger = logging.getLogger()
fhandler = logging.FileHandler(filename="log_noise.log", mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.setLevel(logging.INFO)


def calc_error(mtx1, mtx2):
    cos_theta = (np.trace(mtx1 @ mtx2.T) - 1) / 2
    if cos_theta > 1:
        cos_theta = 1
    if cos_theta < -1:
        cos_theta = -1

    return np.abs(np.arccos(cos_theta) * (180) / np.pi)


def run_noise_test(vol_fname, n_iter, config, signal_noise_ratios, param_setups):
    init_cand = np.eye(3).reshape(1, 3, 3)  # start from identity matrix

    results = {
        "true_mtx": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 3, 3)
        ),
        "optim_mtx_wemd": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 3, 3)
        ),
        "refin_mtx_wemd": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 3, 3)
        ),
        "optim_mtx_l2": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 3, 3)
        ),
        "refin_mtx_l2": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 3, 3)
        ),
        "run_time_wemd": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 2)
        ),
        "run_time_l2": np.zeros(
            (len(signal_noise_ratios), len(param_setups), n_iter, 2)
        ),
    }

    vol_ref = Volume.load(vol_fname)
    vol_ref = center_vol(vol_ref, config)

    for i, snr in enumerate(signal_noise_ratios):
        noise_std = np.sqrt(
            np.linalg.norm(vol_ref._data[0]) ** 2 / (vol_ref._data.shape[1] ** 3 * snr)
        )

        for j, param in enumerate(param_setups):
            config.downsample_res = param[0]
            config.max_iter = param[1]

            for k in range(n_iter):
                # generate random rotation
                true_rot = aspire_Rotation.generate_random_rotations(1)
                results["true_mtx"][i, j, k] = true_rot._matrices[0]
                vol = vol_ref.rotate(true_rot)

                # add noise
                vol = vol + np.random.normal(0, noise_std, vol._data[0].shape).astype(
                    np.float32
                )
                vol_ref_noisy = vol_ref + np.random.normal(
                    0, noise_std, vol_ref._data[0].shape
                ).astype(np.float32)

                # downsample
                vol = vol.downsample(config.downsample_res)
                vol_ref_noisy = vol_ref_noisy.downsample(config.downsample_res)

                # run for wemd
                config.loss_type = "wemd"
                config.corr_length = 0.75

                start_time = time.time()
                opt_rot = run_gaussian_opt(vol, vol_ref_noisy, init_cand, config)
                opt_time = time.time()
                ref_rot = run_nelder_mead_refinement(
                    vol, vol_ref_noisy, opt_rot, config
                )
                end_time = time.time()

                results["run_time_wemd"][i, j, k] = [
                    opt_time - start_time,
                    end_time - start_time,
                ]
                results["optim_mtx_wemd"][i, j, k] = opt_rot
                results["refin_mtx_wemd"][i, j, k] = ref_rot

                # run for l2
                config.loss_type = "l2"
                config.corr_length = 1.0

                start_time = time.time()
                opt_rot = run_gaussian_opt(vol, vol_ref_noisy, init_cand, config)
                opt_time = time.time()
                ref_rot = run_nelder_mead_refinement(
                    vol, vol_ref_noisy, opt_rot, config
                )
                end_time = time.time()

                results["optim_mtx_l2"][i, j, k] = opt_rot
                results["refin_mtx_l2"][i, j, k] = ref_rot
                results["run_time_l2"][i, j, k] = [
                    opt_time - start_time,
                    end_time - start_time,
                ]

                # output results to log
                error_wemd_opt = calc_error(
                    results["optim_mtx_wemd"][i, j, k], results["true_mtx"][i, j, k]
                )
                error_wemd_ref = calc_error(
                    results["refin_mtx_wemd"][i, j, k], results["true_mtx"][i, j, k]
                )
                error_l2_opt = calc_error(
                    results["optim_mtx_l2"][i, j, k], results["true_mtx"][i, j, k]
                )
                error_l2_ref = calc_error(
                    results["refin_mtx_l2"][i, j, k], results["true_mtx"][i, j, k]
                )

                logging.info(
                    f"Results for volume {vol_fname}, snr = {snr}, parameters (downsample_res, max_iter) = {param}, iteration {k}:"
                )
                logging.info(f"True rotation matrix: {results['true_mtx'][i, j, k]}")
                logging.info(
                    f"Optim rotation matrix for wemd: {results['optim_mtx_wemd'][i, j, k]}"
                )
                logging.info(
                    f"Refined rotation matrix for wemd: {results['refin_mtx_wemd'][i, j, k]}"
                )
                logging.info(
                    f"Optim rotation matrix for l2: {results['optim_mtx_l2'][i, j, k]}"
                )
                logging.info(
                    f"Refined rotation matrix for l2: {results['refin_mtx_l2'][i, j, k]}"
                )
                logging.info(f"Error for wemd opt: {error_wemd_opt}")
                logging.info(f"Error for wemd ref: {error_wemd_ref}")
                logging.info(f"Error for l2 opt: {error_l2_opt}")
                logging.info(f"Error for l2 ref: {error_l2_ref}")
                logging.info(
                    f"Time for wemd opt: {results['run_time_wemd'][i, j, k, 0]}"
                )
                logging.info(
                    f"Time for wemd ref: {results['run_time_wemd'][i, j, k, 1]}"
                )
                logging.info(f"Time for l2 opt: {results['run_time_l2'][i, j, k, 0]}")
                logging.info(f"Time for l2 ref: {results['run_time_l2'][i, j, k, 1]}")

    return results


logging.info("Running test for noise")

vol_fname = "volumes/emd_3683.map"

param_setups = [
    [32, 200],
    [64, 150],
]
snrs = [
    1.0 / 32.0,
    1.0 / 128.0
]
n_iter = 25

results = run_noise_test(vol_fname, n_iter, config, snrs, param_setups)

# save results to numpyz file
np.savez("results_for_noise.npz", **results)
