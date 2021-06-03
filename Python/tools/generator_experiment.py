import sys
import datetime
import os
import glob
import tables
import json
import numpy as np

# project imports
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from core.data import open_data_file
from core.model import isensee2017_model
from core.generator import data_generator, get_number_of_steps, get_number_of_patches
from core.training import train_model
from core.metrics import weighted_soft_dice_loss  # , weighted_dice_coefficient_loss
from unet.unet_config import get_kfold_configuration, get_unet_configuration, load_kconfig, unet_model_root_path
from unet.train import kfold_training_and_validation_generators
import nibabel as nib
from core.augment import get_image, scale_image, resample_to_img


def scaling_example(data_file_train):
    # Investigate what scaling of image does
    image = get_image(data_file_train.root.data[0][0], data_file_train.root.affine[0])
    image_scaled_down = resample_to_img(scale_image(image, 0.50), image, interpolation="continuous")
    image_scaled_up = resample_to_img(scale_image(image, 1.50), image, interpolation="continuous")

    image.to_filename(os.path.join('C:\school\RnD', "generator_orig.nii.gz"))
    image_scaled_down.to_filename(os.path.join('C:\school\RnD', "generator_down.nii.gz"))
    image_scaled_up.to_filename(os.path.join('C:\school\RnD', "generator_up.nii.gz"))


def generator_experiment(mconfig, unet_subfolder=None):
    ## DEBUG cheats
    mconfig["patch_shape"] = None

    # open files
    data_path_train = os.path.join(mconfig["kfold_path"],
                                   'config_{}_train.h5'.format(mconfig["kfold_index"]))
    data_path_validation = os.path.join(mconfig["kfold_path"],
                                        'config_{}_validation.h5'.format(mconfig["kfold_index"]))
    print(data_path_train)
    print(mconfig["kfold_path"])
    assert os.path.exists(data_path_train)
    assert os.path.exists(data_path_validation)
    data_file_train = open_data_file(data_path_train)
    data_file_validation = open_data_file(data_path_validation)

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = kfold_training_and_validation_generators(
        data_file_train,
        data_file_validation,
        batch_size=1,
        n_labels=2,
        labels=(0, 1),
        non_healthy=True,
        patch_shape=mconfig["patch_shape"],
        validation_batch_size=mconfig["validation_batch_size"],
        validation_patch_overlap=mconfig["validation_patch_overlap"],
        training_patch_start_offset=mconfig["training_patch_start_offset"],
        permute=True,
        augment=mconfig["augment"],
        skip_blank=mconfig["skip_blank"],
        augment_flip=mconfig["flip"],
        augment_distortion_factor=mconfig["distort"]
    )

    print(train_generator)
    output1 = (next(train_generator))
    print(np.unique(output1[1][0][0], return_counts=True))
    # print(np.unique(output1[1][0][1]))
    # print(np.unique(output1[1][0][2]))
    # print(np.unique(output1[1][0][3]))
    #
    # output1 = (next(train_generator))
    # print(np.unique(output1[1][0][0]))
    # print(np.unique(output1[1][0][1]))
    # print(np.unique(output1[1][0][2]))
    # print(np.unique(output1[1][0][3]))

    # # Investigate what scaling of image does
    # scaling_example(data_file_train)

    # Close datafiles
    data_file_train.close()
    data_file_validation.close()


def main(kfold_subfolder, unet_subfolder):
    # Get configuration
    kconfig = load_kconfig(kfold_subfolder)
    mconfig = get_unet_configuration(kconfig)

    # Train model
    generator_experiment(mconfig, unet_subfolder)


if __name__ == "__main__":
    kfold_subfolder = "experiment05"
    unet_subfolder = "train_exp_05"
    main(kfold_subfolder, unet_subfolder)
