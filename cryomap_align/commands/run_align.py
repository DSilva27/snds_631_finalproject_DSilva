import configargparse
from aspire.volume import Volume
from aspire.utils.rotation import Rotation
import numpy as np
import os
import logging

from cryomap_align.utils import init_config, try_mkdir
from cryomap_align.vol_utils import center_vol
from cryomap_align.gauss_opt_utils import run_gaussian_opt
from cryomap_align.opt_refinement import run_nelder_mead_refinement

parser = configargparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    is_config_file=True,
    help="Config file path",
    required=True,
)
init_config(parser)
config = parser.parse_args()
config.full_save_path = os.path.join(config.save_path, config.experiment_name)

if not try_mkdir(config.full_save_path):
    raise SystemError("Could not create output directory")

# create experiment directory
logging.captureWarnings(True)

logger = logging.getLogger()
fhandler = logging.FileHandler(
    filename=os.path.join(config.full_save_path, "log.log"), mode="a"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.setLevel(logging.INFO)


def main():
    if config.init_cand is None:
        init_cand = np.eye(3).reshape(1, 3, 3)

    else:
        try:
            init_cand = np.load(config.init_cand)
        except FileNotFoundError:
            raise FileNotFoundError("File for initial candidates not found")

    if config.corr_length is None:
        config.corr_length = 0.75  # if config.dist_metric == "wasserstein" else 1.0

    vol = Volume.load(config.vol1)
    vol_ref = Volume.load(config.vol2)

    if config.downsample_res is None:
        config.downsample_res = vol.shape[1]

    assert (
        config.downsample_res <= min(vol.shape[1], vol_ref.shape[1])
    ), "Downsampling resolution must be smaller than volume size"

    vol_cent = center_vol(vol, config)
    vol_ref_cent = center_vol(vol_ref, config)

    vol_cent.save(os.path.join(config.full_save_path, "vol_cent.mrc"), overwrite=True)
    vol_ref_cent.save(os.path.join(config.full_save_path, "vol_ref_cent.mrc"), overwrite=True)

    vol_cent_ds = vol_cent.downsample(config.downsample_res)
    vol_ref_cent_ds = vol_ref_cent.downsample(config.downsample_res)

    opt_rot = run_gaussian_opt(vol_cent_ds, vol_ref_cent_ds, init_cand, config)

    # rotations = np.load(os.path.join(config.full_save_path, "optimized_rotations.npz"))
    # opt_rot = rotations["opt_rot"].astype(np.float32)
    # refined_rot = rotations["refined_rot"].astype(np.float32)

    vol_aligned = vol_cent.rotate(Rotation(opt_rot.T))
    vol_aligned.save(
        os.path.join(config.full_save_path, "vol_aligned.mrc"), overwrite=True
    )

    if config.refine:

        refined_rot = run_nelder_mead_refinement(vol_cent_ds, vol_ref_cent_ds, opt_rot, config)

        vol_aligned = vol_cent.rotate(Rotation(refined_rot.T))
        vol_aligned.save(
            os.path.join(config.full_save_path, "vol_aligned_refined.mrc"),
            overwrite=True,
        )

        np.savez(
            os.path.join(config.full_save_path, "optimized_rotations.npz"),
            opt_rot=opt_rot,
            refined_rot=refined_rot,
        )

    else:
        np.savez(
            os.path.join(config.full_save_path, "optimized_rotations.npz"),
            opt_rot=opt_rot,
        )

    # copy the config file to the output directory
    os.system(f"cp {config.config} {config.save_path}")

    return 0
