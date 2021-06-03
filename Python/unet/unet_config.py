import os
import json

# Constants
prj_root_path = os.path.dirname(os.path.dirname(__file__))
data_root_path = os.path.join(prj_root_path, "data")
kfold_root_path = os.path.join(prj_root_path, "output_kfold")
unet_model_root_path = os.path.join(prj_root_path, 'output_unet')


# Kfold configuration
def get_kfold_configuration(kfold_subfolder):
    kconfig = dict()
    kconfig["data_input"] = os.path.join(data_root_path, "skejby2020_4")
    kconfig["kfold_path"] = os.path.join(kfold_root_path, kfold_subfolder)
    kconfig["image_shape"] = (
        192, 192, 192)  # This determines what shape the images will be cropped/resampled to. (must be divisible by 16)
    kconfig["patch_shape"] = None  # (128, 128, 128)  # switch to None to train on the whole image
    kconfig["n_folds"] = 5  # k

    assert kconfig["n_folds"] >= 3  # we need a training set, validation set and test set

    return kconfig


# Unet model configuration
def get_unet_configuration(kconfig):
    mconfig = kconfig
    mconfig["non_healthy"] = False  # Override with healthy vs. non_healthy labeling
    if not mconfig["non_healthy"]:
        mconfig["label_weights"] =  (1, 1, 1, 1)  # (1.20, 5.91, 3080.77, 5717.39)
        mconfig["labels"] = (0, 1, 2, 3)  # the label numbers on the input image
    else:
        mconfig["label_weights"] = (1, 1)
        mconfig["labels"] = (0, 1)  # the label numbers on the input image
    mconfig["n_labels"] = len(mconfig["labels"])

    mconfig["n_base_filters"] = 16
    mconfig["all_modalities"] = ["HRpQCT"]
    mconfig["training_modalities"] = mconfig[
        "all_modalities"]  # change this if you want to only use some of the modalities
    mconfig["nb_channels"] = len(mconfig["training_modalities"])
    if "patch_shape" in mconfig and mconfig["patch_shape"] is not None:
        mconfig["input_shape"] = tuple([mconfig["nb_channels"]] + list(mconfig["patch_shape"]))
    else:
        mconfig["input_shape"] = tuple([mconfig["nb_channels"]] + list(mconfig["image_shape"]))

    mconfig["truth_channel"] = mconfig["nb_channels"]
    mconfig["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

    mconfig["n_epochs"] = 200  # cutoff the training after this many epochs
    mconfig["kfold_index"] = 0
    mconfig["activation"] = 'softmax'  # 'sigmoid' or 'softmax'
    mconfig["loss"] = 'dl'  # 'dl', 'ce', 'gdl'
    mconfig["initial_learning_rate"] = 5e-4
    mconfig["batch_size"] = 1

    mconfig["validation_batch_size"] = 1
    mconfig["patience"] = 10  # learning rate will be reduced after this many epochs if validation loss is not improving
    mconfig["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    mconfig["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    mconfig["flip"] = False  # augments the data by randomly flipping an axis during
    mconfig["distort"] = None  # True  # switch to None if you want no distortion
    mconfig["augment"] = mconfig["flip"] or mconfig["distort"]
    mconfig["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    mconfig["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
    mconfig["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
    mconfig["skip_blank"] = True  # if True, then patches without any target will be skipped
    mconfig["overwrite"] = True  # If True, will overwrite previous files. If False, will use previously written files.

    mconfig["model_name"] = "model_unet.h5"
    assert mconfig["n_labels"] == len(mconfig["label_weights"])

    return mconfig


def load_kconfig(kfold_subfolder):
    kconfig_path = os.path.join(kfold_root_path, kfold_subfolder, 'kconfig.json')
    with open(kconfig_path) as json_file:
        kconfig = json.load(json_file)
        return kconfig


def load_mconfig(unet_subfolder):
    mconfig_path = os.path.join(unet_model_root_path, unet_subfolder, 'train', 'mconfig.json')
    with open(mconfig_path) as json_file:
        mconfig = json.load(json_file)
        return mconfig


if __name__ == '__main__':
    data_file = 'abc'
    a = get_kfold_configuration(data_file)
    print(a['image_shape'])
