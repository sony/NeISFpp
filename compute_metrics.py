# compute_metrics.py
""" This script calculates & saves evaluation metrics.
Results of Table 1 and Table 2 were computed using this script.

Copyright (c) 2025 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python compute_metrics.py {TARGET_DIR} {SRC_DIR} {EPOCH_NUM} -l {IMAGE_DOMAIN1} {IMAGE_DOMAIN2} ...

    for example:
    $ python compute_metrics.py \
        helmet_neisfpp \  # folder name containing your training result. must be under `results` folder.
        helmet_pp_eval \  # folder name containing GT images to be compared..
        20 \
        -l srgb normal eta k roughness albedo s0 s1 s2

"""
import argparse
from pathlib import Path
from mymodules.evaluationutils import compute_metric

parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the directory including the estimated images.")
parser.add_argument("src_folder", type=str, help="the directory including the ground truths.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("-m", "--use_mask", type=bool, default=True)
parser.add_argument('-l','--image_names', nargs='+', required=True,
                    help='the name list of images. list of allowed names can be found below.')

allowed_name_list = ["srgb", "normal", "eta", "k", "albedo", "roughness", "s0", "s1", "s2"]


if __name__ == '__main__':
    args = parser.parse_args()
    result_folder = args.result_folder
    src_folder = args.src_folder
    epoch_num = args.epoch_num
    use_mask = args.use_mask
    image_names = args.image_names

    # check `image_names`
    for image_name in image_names:
        if image_name not in allowed_name_list:
            raise ValueError(f"unsupported image name: {image_name}")

    target_path = Path("results").joinpath(result_folder, "images_ep{:05d}".format(epoch_num), src_folder)
    src_path = Path("images").joinpath(src_folder)

    if not target_path.exists():
        raise FileNotFoundError("target folder does not exist.")
    if not src_path.exists():
        raise FileNotFoundError("src file does not exist.")

    compute_metric(target_path, src_path, image_names, use_mask)
