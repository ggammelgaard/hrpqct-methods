#!/usr/bin/env python3
""" EXP_04 INDEX 1-4
19	3D-VGGNet	Softmax		SGD_nesterov    lr=1e-3     IDX 1  # Config C3
20	3D-VGGNet	Softmax		SGD_nesterov    lr=1e-3     IDX 2  # Config C3
21	3D-VGGNet	Softmax		SGD_nesterov    lr=1e-3     IDX 3  # Config C3
22	3D-VGGNet	Softmax		SGD_nesterov    lr=1e-3     IDX 4  # Config C3
23	3D-ResNet	Softmax		Adam            lr=27e-6    IDX 1  # Config C7
24	3D-ResNet	Softmax		Adam            lr=27e-6    IDX 2  # Config C7
25	3D-ResNet	Softmax		Adam            lr=27e-6    IDX 3  # Config C7
26	3D-ResNet	Softmax		Adam            lr=27e-6    IDX 4  # Config C7
27	3D-ResNet	Softmax		SGD_nesterov    lr=1e-3     IDX 1  # Config C9
28	3D-ResNet	Softmax		SGD_nesterov    lr=1e-3     IDX 2  # Config C9
29	3D-ResNet	Softmax		SGD_nesterov    lr=1e-3     IDX 3  # Config C9
30	3D-ResNet	Softmax		SGD_nesterov    lr=1e-3     IDX 4  # Config C9
"""
import os
import sys
import datetime
import pandas as pd

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(sys.argv[0])))
    from cnn.cnn_config import get_cnn_configuration, load_cconfig
    from cnn_experiments.experiment_protocol import cnn_experiment_procedure

    # Ensure that variables are entered
    try:
        kfold_subfolder = sys.argv[1]
        experiment_subfolder = sys.argv[2]
    except:
        print("ABORTED. 2 ARGUMENTS NEEDED: 'kfold_subfolder' and 'experiment_subfolder'")
        sys.exit()

    # Get configuration
    cconfig = load_cconfig(kfold_subfolder)
    rconfig = get_cnn_configuration(cconfig)

    # Make test base modifications
    rconfig["n_epochs"] = 150  # cutoff the training after this many epochs
    rconfig["kfold_index"] = 0
    rconfig["batch_size"] = 3
    rconfig["model_type"] = "vggnet"
    rconfig["activation"] = 'softmax'
    rconfig["optimizer"] = 'SGD_nesterov'
    rconfig["initial_learning_rate"] = 1e-3

    rconfig["validation_batch_size"] = 3
    rconfig["patience"] = 10  # learning rate will be reduced after this many epochs if validation loss is not improving
    rconfig["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
    rconfig["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
    rconfig["flip"] = False  # augments the data by randomly flipping an axis during
    rconfig["distort"] = 0.25  # 0.25  # switch to None if you want no distortion
    rconfig["augment"] = rconfig["flip"] or rconfig["distort"]
    rconfig["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    rconfig["skip_blank"] = True  # if True, then patches without any target will be skipped
    rconfig["overwrite"] = True  # If True, will overwrite previous files. If False, will use previously written files.

    # Make individual settings
    settings_dict = dict()
    settings_dict["model_type"] = ('vggnet', 'vggnet', 'vggnet', 'vggnet',
                                   'resnet', 'resnet', 'resnet', 'resnet',
                                   'resnet', 'resnet', 'resnet', 'resnet')
    settings_dict["optimizer"] = ('sgd_nesterov', 'sgd_nesterov', 'sgd_nesterov', 'sgd_nesterov',
                                  'adam', 'adam', 'adam', 'adam',
                                  'sgd_nesterov', 'sgd_nesterov', 'sgd_nesterov', 'sgd_nesterov')
    settings_dict["initial_learning_rate"] = (1e-3, 1e-3, 1e-3, 1e-3,
                                              27e-6, 27e-6, 27e-6, 27e-6,
                                              1e-3, 1e-3, 1e-3, 1e-3)
    settings_dict["kfold_index"] = (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,)

    # Start experiments
    cnn_experiment_procedure(rconfig, settings_dict, experiment_subfolder)
