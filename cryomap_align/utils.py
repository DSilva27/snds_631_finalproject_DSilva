import configargparse
import os
import logging

def try_mkdir(path):
    if not os.path.exists(path):  # path does not exist, create it
        os.makedirs(path)
        return True

    return True


def init_config(parser: configargparse.ArgumentParser):
    parser.add_argument("--map1", type=str, help="Path to map1")
    parser.add_argument("--map2", type=str, help="Path to map2")
    parser.add_argument(
        "--downsample_res", type=int, help="Resolution after downsampling"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations",
        default=200,
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
        type=bool,
        help="Volume might have inverse handedness (used in manifold optimization)",
        default=False,
    )
    # parser.add_argument(
    #     "--dist_metric",
    #     type=str,
    #     help="Distance metric to use (l2 or wasserstein)",
    #     default="l2",
    #     choices=["l2", "wasserstein"],
    # )
    parser.add_argument(
        "--refine",
        type=bool,
        help="Refine alignment using Nelder-Mead algorithm",
        default=False,
    )

    return
