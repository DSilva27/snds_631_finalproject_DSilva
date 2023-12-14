import numpy as np
import pywt
from typing import Optional
from aspire.utils.rotation import Rotation
from aspire.volume import Volume


def vol_to_dwt(
    vol_data: np.ndarray,
    wavelet: Optional[str] = "sym3",
    mode: Optional[str] = "zero",
    level: Optional[int] = 6,
) -> list:
    """
    This function uses PyWavelets to compute the discrete wavelet transform of a 3D volume. The defaults are based on the paper by Shirdhonkar and Jacobs (2008), and Kileel et al. (2021).

    Parameters
    ----------
    vol_data : np.ndarray
        3D volume data.
    wavelet : Optional[str], optional
        Wavelet to use. The default is "sym3".
    mode : Optional[str], optional
        Mode to use. The default is "zero".
    level : Optional[int], optional
        Level of decomposition. The default is 6.

    Returns
    -------
    dwt_coeffs : list
        List of wavelet coefficients. The first element is the approximation coefficients, and the remaining elements are the detail coefficients.

    """
    assert vol_data.ndim == 3, "Volume must be 3D"

    dwt_coeffs = pywt.wavedec(
        vol_data / vol_data.sum(), wavelet=wavelet, mode=mode, level=level
    )

    return dwt_coeffs


def earthmovers_dist(vol1_dwt: list, vol2_dwt: list) -> float:
    """
    Calculate the Earthmover's distance between two volumes.

    Parameters
    ----------
    vol1_dwt : list
        Wavelet coefficients of the first volume.
    vol2_dwt : list
        Wavelet coefficients of the second volume.

    Returns
    -------
    earthm_dist : float
        Earthmover's distance between the two volumes.
    """

    assert len(vol1_dwt) == len(
        vol2_dwt
    ), "Volumes Wvlet transf. must have the same number of levels"

    earthm_dist = 0
    for i in range(len(vol1_dwt)):
        earthm_dist += np.linalg.norm(
            vol1_dwt[i].flatten() - vol2_dwt[i].flatten(), ord=1
        ) * 2 ** (-i * 5 / 2)

    return earthm_dist


def calc_wemd(rot: np.ndarray, vol_obj: Volume, vol_ref_dwt: list) -> float:
    """
    Calculate the loss between the objective volume (volume we wish to align) and reference volume. The loss is the Earthmover's distance between the wavelet coefficients of the objective volume and the reference volume.

    Parameters
    ----------
    rot : np.ndarray
        Rotation matrix.
    vol_obj : Volume
        Objective volume.
    vol_ref_dwt : list
        Reference volume wavelet coefficients.

    Returns
    -------
    loss : float
        Loss between the objective volume and reference volume.
    """

    vol_obj_rot = vol_obj.rotate(Rotation(rot.T))._data[0]

    vol_obj_dwt = vol_to_dwt(vol_obj_rot)
    loss = earthmovers_dist(vol_obj_dwt, vol_ref_dwt)

    return loss
