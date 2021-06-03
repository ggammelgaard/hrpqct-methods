import os
import json
import datetime
import wandb  # resoure utilization monitoring
from cnn.cnn_config import get_cnn_configuration, load_cconfig, cnn_model_root_path
from core.model.tuber import tuber_model
from core.data import open_data_file
from cnn.cnn_generator import cnn_training_and_validation_generators
from core.training import train_model
from core.model.vggnet import vggnet_model
from core.model.resnet import resnet_model

from keras import backend as K

K.set_image_dim_ordering('th')  # channels_first


def train(rconfig, cnn_subfolder=None):
    # Create output folder
    if not cnn_subfolder:
        cnn_subfolder = datetime.datetime.now().strftime("%y%m%d_%H%M")
    train_subfolder_path = os.path.join(cnn_model_root_path, cnn_subfolder, 'train')
    try:
        os.makedirs(train_subfolder_path)
    except:
        pass

    # Dump cnn_config for this run
    with open(os.path.join(train_subfolder_path, 'rconfig.json'), 'w') as json_file:
        json.dump(rconfig, json_file, sort_keys=True, indent=4)

    # Dump wandb data for this run
    wandb.init(project="CNN", dir=train_subfolder_path, name=cnn_subfolder)

    # instantiate new model
    if rconfig["model_type"] == 'vggnet':
        model_func = vggnet_model
    elif rconfig["model_type"] == 'tuber':
        model_func = tuber_model
    elif rconfig["model_type"] == 'resnet':
        model_func = resnet_model
    else:
        print("ERROR! INOCRRECT MODEL_TYPE IN RCONFIG.")
        return
    model_file = os.path.join(train_subfolder_path, rconfig["model_name"])
    model = model_func(input_shape=rconfig["input_shape"], initial_learning_rate=rconfig["initial_learning_rate"],
                       model_name=model_file, activation=rconfig["activation"], optimizer=rconfig["optimizer"])

    # open files
    data_path_train = os.path.join(rconfig["kfold_path"],
                                   'config_{}_train_cc.h5'.format(rconfig["kfold_index"]))
    data_path_validation = os.path.join(rconfig["kfold_path"],
                                        'config_{}_validation_cc.h5'.format(rconfig["kfold_index"]))
    print(data_path_train)
    print(rconfig["kfold_path"])
    assert os.path.exists(data_path_train)
    assert os.path.exists(data_path_validation)
    data_file_train = open_data_file(data_path_train)
    data_file_validation = open_data_file(data_path_validation)

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = cnn_training_and_validation_generators(
        data_file_train,
        data_file_validation,
        batch_size=rconfig["batch_size"],
        validation_batch_size=rconfig["validation_batch_size"],
        permute=rconfig["permute"],  # should be enabled when i use cubic metrics
        augment=rconfig["augment"],
        skip_blank=rconfig["skip_blank"],
        augment_flip=rconfig["flip"],
        augment_distortion_factor=rconfig["distort"])

    # run training
    train_model(model=model,
                model_file=model_file,
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=rconfig["initial_learning_rate"],
                learning_rate_drop=rconfig["learning_rate_drop"],
                learning_rate_patience=rconfig["patience"],
                early_stopping_patience=rconfig["early_stop"],
                n_epochs=rconfig["n_epochs"],
                train_output_folder=train_subfolder_path)

    # close data files
    data_file_train.close()
    data_file_validation.close()

    # # post training
    # model.save(os.path.splitext(model_file)[0] + '_last_epoch.5')
    return cnn_subfolder


def main(kfold_subfolder, cnn_subfolder):
    # Get configuration
    cconfig = load_cconfig(kfold_subfolder)
    rconfig = get_cnn_configuration(cconfig)

    # Train model
    train(rconfig, cnn_subfolder)


if __name__ == "__main__":
    kfold_subfolder = "210226_1023"
    cnn_subfolder = "cnn_exp_01"
    main(kfold_subfolder, cnn_subfolder)
