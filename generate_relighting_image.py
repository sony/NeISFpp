# generate_relighting_image.py
""" Run this script for training.

Copyright (c) 2025 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python generate_relighting_image.py \
        {YOUR RESULT FOLDER} \
        {TARGET FOLDER} \
        {EPOCH NUM} \
        {ENV MAP NAME} \
        -b {B SIZE} \
        -l {SAMPLE ILLUM NUM}
"""

import argparse

from mymodules.trainers import trainer_provider
from mymodules.trainers.trainers_base import SAVED_CONFIG_NAME, RESULT_PARENT_PATH


parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the folder name your models are stored.")
parser.add_argument("image_folder", type=str, help="the folder name your images are stored.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("env_map_name", type=str, help="the name of environment map.")
parser.add_argument("--render_gt", action='store_true')
parser.add_argument("-b", "--batch_size", type=int, default=2048)
parser.add_argument("-l", "--light_num", type=int, default=10000)



if __name__ == "__main__":
    args = parser.parse_args()
    result_folder = args.result_folder
    image_folder = args.image_folder
    epoch_num = args.epoch_num
    env_map_name = args.env_map_name
    render_gt = args.render_gt
    batch_size = args.batch_size
    light_num = args.light_num

    config_path = RESULT_PARENT_PATH.joinpath(result_folder, SAVED_CONFIG_NAME)

    trainer = trainer_provider(config_path, is_train=False, inference_folder=image_folder, inference_batch=batch_size)
    trainer.render_relighting(image_folder, epoch_num, env_map_name, light_num, render_gt)
