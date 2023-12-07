import numpy as np
import logging

from scipy.spatial.transform import Rotation as scipy_rotation
from aspire.utils.rotation import Rotation as aspire_rotation
from scipy.optimize import minimize


def refinm_loss(quat, vol1, vol2):
    rot = scipy_rotation.from_quat(quat).as_matrix()
    rot = aspire_rotation(rot.astype(np.float32))

    vol_obj_rot = vol1.rotate(rot)

    return np.linalg.norm(vol_obj_rot - vol2)


def run_nelder_mead_refinement(vol, vol_ref, init_rot, config):
    quat_0 = scipy_rotation.from_matrix(init_rot).as_quat()
    result = minimize(
        refinm_loss,
        quat_0,
        args=(vol_ref, vol),
        method="nelder-mead",
        options={"disp": True},
    )
    quat_f = result.x
    refined_rot = scipy_rotation.from_quat(quat_f).as_matrix()

    return refined_rot.astype(np.float32)
