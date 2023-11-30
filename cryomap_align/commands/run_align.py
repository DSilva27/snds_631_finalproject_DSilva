import configargparse
from aspire.volume import Volume
import numpy as np
import os
import logging

from cryomap_align.utils import init_config, try_mkdir
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

if not try_mkdir(config.save_path):
    raise SystemError("Could not create output directory")

# create experiment directory

logger = logging.getLogger()
fhandler = logging.FileHandler(
    filename=os.path.join(config.save_path, "log.log"), mode="a"
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

    map1 = Volume.load(config.map1)
    map2 = Volume.load(config.map2)

    opt_rot = run_gaussian_opt(map1, map2, init_cand, config)

    if config.refine:
        refined_rot = run_nelder_mead_refinement(opt_rot, map1, map2, config)

        np.savez(
            os.path.join(config.save_path, "optimized_rotations.npz"),
            opt_rot=opt_rot,
            refined_rot=refined_rot,
        )

    else:
        np.savez(
            os.path.join(config.save_path, "optimized_rotations.npz"),
            opt_rot=opt_rot,
       )
        
    # copy the config file to the output directory
    os.system(f"cp {config.config} {config.save_path}")
    
    return 0
