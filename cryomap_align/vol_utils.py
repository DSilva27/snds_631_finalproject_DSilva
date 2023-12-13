import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mrcfile
import logging
from scipy import ndimage
from aspire.volume import Volume


def get_atomic_number_(atom_name):
    atomic_numbers = {
        "H": 1.0,
        "HO": 1.0,
        "W": 18.0,
        "C": 6.0,
        "A": 7.0,
        "N": 7.0,
        "O": 8.0,
        "P": 15.0,
        "K": 19.0,
        "S": 16.0,
        "AU": 79.0,
    }

    return atomic_numbers[atom_name]


def parse_pdb(filename):
    assert filename.endswith(".pdb"), "File must be a PDB file"

    univ = mda.Universe(filename)
    univ.atoms.translate(-univ.atoms.center_of_mass())

    atomic_model = np.zeros((5, univ.select_atoms("not name H*").n_atoms))
    atomic_model[0:3, :] = univ.select_atoms("not name H*").positions.T
    atomic_model[3, :] = np.array(
        [get_atomic_number_(x) for x in univ.select_atoms("not name H*").elements]
    )
    atomic_model[4, :] = (1.0 / np.pi) ** 2

    return atomic_model


def generate_grid1d(box_size: int, pixel_size: float = 1):
    grid_limit = pixel_size * box_size * 0.5
    grid = np.arange(-grid_limit, grid_limit, pixel_size)[0:box_size]

    return grid


def center_vol(volume: Volume, config: configargparse.Namespace):
    """
    Center the vol by shifting the center of mass to the origin.

    Parameters
    ----------
    volume : aspire.volume.Volume
        Volume to be centered.
    config : configargparse.Namespace
        Configuration object (from config file).

    Returns
    -------
    vol_centered : aspire.volume.Volume
        Centered volume.
    """

    logging.info("Centering volume")
    vol_copy = np.copy(volume._data[0])
    vol_copy[vol_copy < config.centering_noise_filter] = 0.0

    vol_copy = vol_copy / vol_copy.sum()
    grid = generate_grid1d(vol_copy.shape[-1], config.pixel_size)

    center_of_mass = np.zeros(3)
    for i, ax in enumerate([(1, 2), (0, 2), (0, 1)]):
        center_of_mass[i] = np.sum(vol_copy, axis=ax) @ grid

    logging.info("Center of mass: %s", center_of_mass)

    vol_cent_data = ndimage.shift(
        vol_copy, -center_of_mass, order=config.center_order, mode="constant"
    )
    vol_centered = Volume(vol_cent_data[np.newaxis, ...])

    logging.info("Volume centered")

    return vol_centered


def generate_vol_(atomic_model, grid, res):
    gauss_var = atomic_model[4, :] * res**2
    gauss_amp = atomic_model[3, :] / np.sqrt(gauss_var * 2.0 * np.pi)

    gauss_x = gauss_amp * np.exp(
        -0.5 * (grid[:, None] - atomic_model[0, :]) ** 2 / gauss_var
    )
    gauss_y = gauss_amp * np.exp(
        -0.5 * (grid[:, None] - atomic_model[1, :]) ** 2 / gauss_var
    )

    gauss_z = gauss_amp * np.exp(
        -0.5 * (grid[:, None] - atomic_model[2, :]) ** 2 / gauss_var
    )

    volume = np.einsum("ji, ki, li->jkl", gauss_z, gauss_y, gauss_x)

    return volume


def generate_volume(atomic_model, box_size, pixel_size, res):
    grid = generate_grid(box_size, pixel_size)
    volume = generate_vol_(atomic_model, grid, res)

    return volume


def pdb_to_mrc(filename, rot_mtx, box_size, pixel_size, res=None, outfile=None):
    if res is None:
        res = pixel_size * 2

    if outfile is None:
        outfile = filename.split(".")[0] + ".mrc"

    atomic_model = parse_pdb(filename)
    atomic_model[:4, :] = rot_mtx @ atomic_model[:4, :]

    volume = generate_volume(atomic_model, box_size, pixel_size, res)

    return Volume(volume)
