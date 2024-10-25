"""
Wrapper functions for a COLMAP-free version of the hloc library.
"""

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R

from .read_write_model import Image, Reconstruction


def get_reg_image_ids(reconstruction):
    if isinstance(reconstruction, Reconstruction):
        return [img.id for (_, img) in reconstruction.images.items()]
    elif isinstance(reconstruction, pycolmap.Reconstruction):
        return reconstruction.reg_image_ids()
    else:
        raise ValueError("Unknown reconstruction type.")


def get_keypoints(reconstruction, image):
    if isinstance(image, Image):
        keypoints = np.array([xy for xy in image.xys])
        point_ids = [id for id in image.point3D_ids]
        visible = np.array(
            [len(reconstruction.points3D[id].image_ids) > 0 for id in point_ids]
        )
        return keypoints, visible
    elif isinstance(image, pycolmap.Image):
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])
        return keypoints, visible
    else:
        raise ValueError("Unknown reconstruction type.")


def get_track_lengths(reconstruction, image):
    if isinstance(reconstruction, Reconstruction):
        point_ids = [id for id in image.point3D_ids]
        return np.array(
            [len(reconstruction.points3D[id].image_ids) for id in point_ids]
        )
    elif isinstance(reconstruction, pycolmap.Reconstruction):
        tl = np.array(
            [
                (
                    reconstruction.points3D[p.point3D_id].track.length()
                    if p.has_point3D()
                    else 1
                )
                for p in image.points2D
            ]
        )
        return tl
    else:
        raise ValueError("Unknown reconstruction type.")


def get_depths(reconstruction, image):
    if isinstance(reconstruction, Reconstruction):
        point_ids = [id for id in image.point3D_ids]
        r = R.from_quat(image.qvec, scalar_first=True)
        z = np.array(
            [
                (r.as_matrix().dot(reconstruction.points3D[id].xyz))[-1]
                for id in point_ids
            ]
        )
        return z
    elif isinstance(reconstruction, pycolmap.Reconstruction):
        p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        z = np.array(
            [(image.cam_from_world * reconstruction.points3D[j].xyz)[-1] for j in p3ids]
        )
        return z
    else:
        raise ValueError("Unknown reconstruction type.")
