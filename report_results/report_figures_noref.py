import configargparse
from aspire.volume import Volume
from aspire.utils.rotation import Rotation as aspire_Rotation
from scipy.spatial.transform import Rotation as scipy_Rotation
import numpy as np
import os
import logging
import time
import sys

from cryomap_align.utils import init_config, try_mkdir, center_vol
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
fhandler = logging.FileHandler(
    filename="log_no_refinment.log", mode="a"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.setLevel(logging.INFO)


def run_align_test(vol_fnames, n_iter, config, param_setups):

    init_cand = np.eye(3).reshape(1, 3, 3) # start from identity matrix

    results = {
        "true_quats": np.zeros((len(vol_fnames), len(param_setups), n_iter, 4)),
        "optim_quats_wemd": np.zeros((len(vol_fnames), len(param_setups), n_iter, 4)),
        "optim_quats_l2": np.zeros((len(vol_fnames), len(param_setups), n_iter, 4)),
        "run_time_wemd": np.zeros((len(vol_fnames), len(param_setups), n_iter, 1)),
        "run_time_l2": np.zeros((len(vol_fnames), len(param_setups), n_iter, 1)),
    }

    for i, vol_fname in enumerate(vol_fnames):

        vol_ref = Volume.load(vol_fname)
        vol_ref = center_vol(vol_ref, config)
        
        for j, param in enumerate(param_setups):
            
            config.downsample_res = param[0]
            config.max_iter = param[1]
            vol_ref_ds = vol_ref.downsample(config.downsample_res)

            for k in range(n_iter):
                
                # generate random rotation
                true_rot = aspire_Rotation.generate_random_rotations(1)
                results["true_quats"][i, j, k] = scipy_Rotation.from_matrix(true_rot._matrices[0]).as_quat()
                vol = vol_ref.rotate(true_rot).downsample(config.downsample_res)

                # run for wemd
                config.loss_type = "wemd"
                config.corr_length = 0.75
                
                start_time = time.time()
                opt_rot = run_gaussian_opt(vol, vol_ref_ds, init_cand, config)
                opt_time = time.time()

                results["run_time_wemd"][i, j, k] = [opt_time - start_time]
                results["optim_quats_wemd"][i, j, k] = scipy_Rotation.from_matrix(opt_rot).as_quat()

                # run for l2
                config.loss_type = "l2"
                config.corr_length = 1.0

                start_time = time.time()
                opt_rot = run_gaussian_opt(vol, vol_ref_ds, init_cand, config)
                opt_time = time.time()

                results["optim_quats_l2"][i, j, k] = scipy_Rotation.from_matrix(opt_rot).as_quat()
                results["run_time_l2"][i, j, k] = [opt_time - start_time]


                # output results to log
                err_wemd_opt = np.linalg.norm(results["optim_quats_wemd"][i, j, k] - results["true_quats"][i, j, k]) / np.linalg.norm(results["true_quats"][i, j, k])

                err_l2_opt = np.linalg.norm(results["optim_quats_l2"][i, j, k] - results["true_quats"][i, j, k]) / np.linalg.norm(results["true_quats"][i, j, k])

                logging.info(f"Results for volume {vol_fname}, parameters (downsample_res, max_iter) = {param}, iteration {k}:")
                logging.info(f"Error for wemd opt: {err_wemd_opt}")
                logging.info(f"Error for l2 opt: {err_l2_opt}")
                logging.info(f"Time for wemd opt: {results['run_time_wemd'][i, j, k, 0]}")
                logging.info(f"Time for l2 opt: {results['run_time_l2'][i, j, k, 0]}")

    return results

logging.info("Running test for no refinement")

vol_fnames = [
    "volumes/emd_3683.map.gz",
    "volumes/emd_25892.map.gz",
    "volumes/emd_9515.map.gz",
    "volumes/emd_23006.map.gz",
]
param_setups = [
    [32, 150],
    [32, 200],
    [64, 150],
    [64, 200],
]
n_iter = 50

results = run_align_test(vol_fnames, n_iter, config, param_setups)

# save results to numpyz file
np.savez(
    os.path.join(config.full_save_path, "results_no_refinement.npz"),
    **results
)