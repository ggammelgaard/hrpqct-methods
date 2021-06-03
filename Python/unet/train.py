import datetime
import os
import glob
import tables
import json
import numpy as np

from core.data import open_data_file
from core.model import isensee2017_model
from core.generator import data_generator, get_number_of_steps, get_number_of_patches
from core.training import train_model
from unet.unet_config import get_kfold_configuration, get_unet_configuration, load_kconfig, unet_model_root_path
from core.metrics import update_global_weights
import wandb  # resoure utilization monitoring


def kfold_training_and_validation_generators(data_train_file, data_validation_file, batch_size, n_labels,
                                             labels=None, augment=False, non_healthy=False,
                                             augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                             validation_patch_overlap=0, training_patch_start_offset=None,
                                             validation_batch_size=None, skip_blank=True, permute=False):
    training_list = list(
        range(data_train_file.root.data.shape[0]))  # arbitrary list needed for the data generator function
    validation_list = list(
        range(data_validation_file.root.data.shape[0]))  # arbitrary list needed for the datagenerator function

    training_generator = data_generator(data_train_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,
                                        non_healthy=non_healthy)

    validation_generator = data_generator(data_validation_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          non_healthy=non_healthy)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(get_number_of_patches(data_train_file, training_list, patch_shape,
                                                                   skip_blank=skip_blank,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(get_number_of_patches(data_validation_file, validation_list, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def train(mconfig, unet_subfolder=None):
    # Create output folder
    if not unet_subfolder:
        unet_subfolder = datetime.datetime.now().strftime("%y%m%d_%H%M")
    train_subfolder_path = os.path.join(unet_model_root_path, unet_subfolder, 'train')
    try:
        os.makedirs(train_subfolder_path)
    except:
        pass

    # Update labels if HNH
    if mconfig["non_healthy"]:
        mconfig["labels"] = (0, 1)
        mconfig["n_labels"] = len(mconfig["labels"])

    # Dump unet_config for this run
    with open(os.path.join(train_subfolder_path, 'mconfig.json'), 'w') as json_file:
        json.dump(mconfig, json_file, sort_keys=True, indent=4)

    # Dump wandb data for this run
    wandb.init(project="UNET", dir=train_subfolder_path, name=unet_subfolder)

    # Update weights used for training metrics
    update_global_weights(mconfig)

    # instantiate new model
    model_file = os.path.join(train_subfolder_path, mconfig["model_name"])
    model = isensee2017_model(input_shape=mconfig["input_shape"], n_labels=mconfig["n_labels"],
                              initial_learning_rate=mconfig["initial_learning_rate"],
                              n_base_filters=mconfig["n_base_filters"], model_name=model_file,
                              activation=mconfig["activation"], loss=mconfig["loss"],
                              loss_weights=mconfig["label_weights"])

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
        batch_size=mconfig["batch_size"],
        n_labels=mconfig["n_labels"],
        labels=mconfig["labels"],
        non_healthy=mconfig["non_healthy"],
        patch_shape=mconfig["patch_shape"],
        validation_batch_size=mconfig["validation_batch_size"],
        validation_patch_overlap=mconfig["validation_patch_overlap"],
        training_patch_start_offset=mconfig["training_patch_start_offset"],
        permute=mconfig["permute"],
        augment=mconfig["augment"],
        skip_blank=mconfig["skip_blank"],
        augment_flip=mconfig["flip"],
        augment_distortion_factor=mconfig["distort"])

    # run training
    train_model(model=model,
                model_file=model_file,
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=mconfig["initial_learning_rate"],
                learning_rate_drop=mconfig["learning_rate_drop"],
                learning_rate_patience=mconfig["patience"],
                early_stopping_patience=mconfig["early_stop"],
                n_epochs=mconfig["n_epochs"],
                train_output_folder=train_subfolder_path)

    # close data files
    data_file_train.close()
    data_file_validation.close()

    # # post training
    # model.save(os.path.splitext(model_file)[0] + '_last_epoch.5')
    return unet_subfolder


def main(kfold_subfolder, unet_subfolder):
    # Get configuration
    kconfig = load_kconfig(kfold_subfolder)
    mconfig = get_unet_configuration(kconfig)

    # Train model
    train(mconfig, unet_subfolder)


if __name__ == "__main__":
    kfold_subfolder = "experiment05"
    unet_subfolder = "train_exp_05"
    main(kfold_subfolder, unet_subfolder)
