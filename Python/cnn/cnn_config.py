import os
import json
from unet.unet_config import prj_root_path, kfold_root_path

# Constants
cnn_model_root_path = os.path.join(prj_root_path, 'output_cnn')


# CC conversion configuration
def get_cc_configuration(kconfig):
    cconfig = kconfig
    cconfig["cc_shape"] = (110, 110, 110)
    return cconfig


# CNN model configuration
def get_cnn_configuration(cconfig):
    rconfig = cconfig

    rconfig["n_epochs"] = 150  # cutoff the training after this many epochs
    rconfig["kfold_index"] = 0
    rconfig["batch_size"] = 3  # 5
    rconfig["model_type"] = "tuber"  # "resnet"  # "vggnet"
    rconfig["activation"] = 'softmax'  # 'sigmoid'
    rconfig["optimizer"] = 'adam'  # sgd_nesterov  # sgd_momentum
    rconfig["initial_learning_rate"] = 1e-5  # 0.001  # 27e-6

    rconfig["validation_batch_size"] = 3  # 10
    rconfig["patience"] = 10  # learning rate will be reduced after this many epochs if validation loss is not improving
    rconfig["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    rconfig["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    rconfig["flip"] = False  # augments the data by randomly flipping an axis during
    rconfig["distort"] = 0.25  # 0.25  # switch to None if you want no distortion
    rconfig["augment"] = rconfig["flip"] or rconfig["distort"]
    rconfig["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    rconfig["skip_blank"] = True  # if True, then patches without any target will be skipped
    rconfig["overwrite"] = True  # If True, will overwrite previous files. If False, will use previously written files.

    rconfig["model_name"] = "model_cnn.h5"
    rconfig["input_shape"] = tuple([1] + rconfig["cc_shape"])

    return rconfig


def load_cconfig(kfold_subfolder):
    cconfig_path = os.path.join(kfold_root_path, kfold_subfolder, 'cconfig.json')
    with open(cconfig_path) as json_file:
        cconfig = json.load(json_file)
        return cconfig


def load_rconfig(unet_subfolder):
    rconfig_path = os.path.join(cnn_model_root_path, unet_subfolder, 'train', 'rconfig.json')
    with open(rconfig_path) as json_file:
        rconfig = json.load(json_file)
        return rconfig
