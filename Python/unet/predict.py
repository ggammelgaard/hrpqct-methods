import os
import datetime
import sys

from core.prediction import run_validation_cases
from unet.unet_config import unet_model_root_path, load_mconfig
from core.data import open_data_file
from core.metrics import update_global_weights


def predict(mconfig, unet_subfolder):
    # Define paths
    train_subfolder_path = os.path.join(unet_model_root_path, unet_subfolder, 'train')
    output_folder = os.path.join(unet_model_root_path, unet_subfolder, 'predict')
    try:
        os.makedirs(output_folder)
    except:
        pass

    model_file_path = os.path.join(train_subfolder_path, mconfig["model_name"])
    # Update weights used for training metrics
    update_global_weights(mconfig)

    # open file
    data_path_test = os.path.join(mconfig['kfold_path'], 'config_{}_test.h5'.format(mconfig["kfold_index"]))
    assert os.path.exists(data_path_test)
    data_file_test = open_data_file(data_path_test)
    index_list = list(
        range(data_file_test.root.data.shape[0]))  # arbitrary list needed for the run_validation_cases function

    # this is the test cases
    run_validation_cases(validation_keys_file=index_list,
                         model_file=model_file_path,
                         training_modalities=mconfig["training_modalities"],
                         labels=mconfig["labels"],
                         hdf5_file=data_path_test,
                         output_label_map=True,
                         output_dir=output_folder,
                         activation=mconfig['activation'],
                         non_healthy=mconfig['non_healthy'])

    data_file_test.close()  # close file again


def main(unet_subfolder):
    # Get configuration
    mconfig = load_mconfig(unet_subfolder)

    # Train model
    predict(mconfig, unet_subfolder)


if __name__ == "__main__":
    # unet_subfolder = "20210217_1213"
    unet_subfolder = "210427_1428"
    main(unet_subfolder)
