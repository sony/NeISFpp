# evaluationutils.py
""" Library for evaluation.

Copyright (c) 2025 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np
import pathlib
import json
import os
from mymodules.imageutils import my_read_image, MAX_16BIT


def calc_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute PSNR between img1 and img2.
    Note that images should be normalized into [0, 1].

    Args:
        img1 (np.ndarray): image (h, w, 3).
        img2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated PSNR.
    """

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if img1.shape[2] != 3 or img2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")
    if np.max(img1) > 1 or np.max(img2) > 1:
        raise ValueError("img must be normalized into [0, 1].")
    if np.min(img1) < 0 or np.min(img2) < 0:
        raise ValueError("img must be normalized into [0, 1].")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    diff = img1 - img2  # (h, w, 3)

    if mask is not None:
        diff = diff[mask]  # (n,)

    return float(10 * np.log10(1 / np.mean(diff * diff)))


def calc_l1(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute mean absolute error between img1 and img2.

    Args:
        img1 (np.ndarray): image (h, w, 3).
        img2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated absolute error.
    """

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if img1.shape[2] != 3 or img2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    diff = np.abs(img1 - img2)  # (h, w, 3)

    if mask is not None:
        diff = diff[mask]

    return float(diff.mean())


def calc_mae(normal1: np.ndarray, normal2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute mean angular error (MAE) between normal1 and normal2.
    Note that images should be normalized into [-1, 1].

    Args:
        normal1 (np.ndarray): image (h, w, 3).
        normal2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated MEA [rad].
    """

    if normal1.ndim != 3 or normal2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if normal1.shape[2] != 3 or normal2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")
    if np.max(normal1) > 1 or np.max(normal2) > 1:
        raise ValueError("img must be normalized into [-1, 1].")
    if np.min(normal1) < -1 or np.min(normal2) < -1:
        raise ValueError("img must be normalized into [-1, 1].")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    normal1 /= (np.linalg.norm(normal1, axis=2, keepdims=True) + 1e-07)
    normal2 /= (np.linalg.norm(normal2, axis=2, keepdims=True) + 1e-07)
    dot = np.sum(normal1 * normal2, axis=2)  # (h, w)

    if mask is not None:
        dot = dot[mask[..., 0]]

    dot = np.clip(dot, -1, 1)

    theta = np.arccos(dot)

    return float(np.rad2deg(np.mean(theta)))


def align_scale(img: np.ndarray, img_gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Align the scale of the target image for scaling-invariant metric.
    Refer to: https://arxiv.org/pdf/2104.00674.pdf Eq.11

    Args:
        img (np.array): the estimated image of shape (h, w, 3).
        img_gt (np.array): the ground truth image of shape (h, w, 3).
        mask (np.array): object mask of shape (h, w, 3).

    Returns:
        (np.array): the scaled image of shape (h, w, 3).
    """

    scale_map = img_gt / (img + 1e-07)

    scale_r = np.median(scale_map[:, :, 0][mask[:, :, 0]])
    scale_g = np.median(scale_map[:, :, 1][mask[:, :, 1]])
    scale_b = np.median(scale_map[:, :, 2][mask[:, :, 2]])

    img = np.concatenate([img[:, :, 0:1] * scale_r, img[:, :, 1:2] * scale_g, img[:, :, 2:3] * scale_b], axis=-1)

    return img

def compute_metric(est_path: pathlib.Path, gt_path:pathlib.Path, image_names: list, use_mask: bool=True) -> None:
    """ this function computes & save the metrics between the estimated images and gt images. 

    Args:
        est_path (pathlib.Path): the folder of estimated images.
        gt_path (pathlib.Path): the folder of ground-truth images.
        image_names (list): a list of parameters that need to compute metric.
        use_mask (bool): compute the masked-out metric or not?
    """      
    
    num_images = len(os.listdir(str(gt_path.joinpath("images"))))
    err_dicts = {image_name: 0. for image_name in image_names}
    for ii in range(num_images):
        if use_mask:
            mask_path = gt_path.joinpath("masks").joinpath("img_{:03d}.png".format(ii+1))
            mask_img = my_read_image(mask_path) / MAX_16BIT
            mask_img = (mask_img > 0.999)

            c_mask_path = gt_path.joinpath("c_masks").joinpath("img_{:03d}.png".format(ii+1))
            c_mask_img = my_read_image(c_mask_path) / MAX_16BIT
        else:
            mask_img = None
        err_dict = {}
        for image_name in image_names:
            if image_name in ["s0", "s1", "s2"]:
                gt_img_path = gt_path.joinpath(f"images_{image_name}", "img_{:03d}.exr".format(ii+1))
                est_img_path = est_path.joinpath("{:03d}_{}.exr".format(ii+1, image_name))
                gt_img = my_read_image(gt_img_path)
                est_img = my_read_image(est_img_path) 
            elif image_name in ["eta", "k"]:
                gt_img_path = gt_path.joinpath(f"{image_name}s", "img_{:03d}.exr".format(ii+1))
                est_img_path = est_path.joinpath("{:03d}_{}.exr".format(ii+1, image_name))
                gt_img = my_read_image(gt_img_path)
                est_img = my_read_image(est_img_path) 
            elif image_name in ["albedo", "roughness"]:
                gt_img_path = gt_path.joinpath(f"{image_name}s", "img_{:03d}.png".format(ii+1))
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, image_name))
                gt_img = my_read_image(gt_img_path) / MAX_16BIT
                est_img = my_read_image(est_img_path) / MAX_16BIT
            elif image_name == "srgb":
                gt_img_path = gt_path.joinpath("images", "img_{:03d}.png".format(ii+1))
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, image_name))
                gt_img = my_read_image(gt_img_path) / MAX_16BIT
                est_img = my_read_image(est_img_path) / MAX_16BIT
            elif image_name == "normal":
                gt_img_path = gt_path.joinpath(f"{image_name}s", "img_{:03d}.png".format(ii+1))
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, image_name))
                gt_img = (my_read_image(gt_img_path) / MAX_16BIT) * 2. - 1.
                est_img = (my_read_image(est_img_path) / MAX_16BIT) * 2 -1
            else:
                raise ValueError("unknown image name.")

            if image_name == "normal":
                metric = calc_mae(gt_img, est_img, mask_img)
            elif image_name in ["s0", "s1", "s2"]:
                metric = calc_l1(gt_img, est_img, mask_img)
            elif image_name == "roughness":
                metric = calc_l1(gt_img, est_img, mask_img)
            elif image_name == "albedo":
                metric = calc_l1(gt_img, est_img, mask_img & (c_mask_img<0.001))
            elif image_name in ["eta", "k"]:
                metric = calc_l1(gt_img, est_img, mask_img & (c_mask_img>0.999))
            else:
                metric = calc_psnr(gt_img, est_img, mask_img)

            err_dict[image_name] = metric
            err_dicts[image_name] += metric / num_images
        with open(est_path.joinpath("{:03d}.json".format(ii+1)), "w") as outfile:
            json.dump(err_dict, outfile)
    # save the average for all test images
    with open(est_path.joinpath("metric.json"), "w") as outfile:
        json.dump(err_dicts, outfile)
