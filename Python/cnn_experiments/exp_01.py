#!/usr/bin/env python3
""" EXP_01 INDEX 0
01	3D-VGGNet	Softmax		Adam            lr=27e-6    IDX 0   # Config C1
02	3D-VGGNet	Sigmoid		Adam            lr=27e-6    IDX 0   # Config C2
03	3D-VGGNet	Softmax		SGD_nesterov    lr=1e-3     IDX 0   # Config C3
04	3D-VGGNet	Sigmoid		SGD_nesterov    lr=1e-3     IDX 0   # Config C4
05	3D-VGGNet	Softmax		SGD_momentum    lr=1e-6     IDX 0   # Config C5
06	3D-VGGNet	Sigmoid		SGD_momentum    lr=1e-6     IDX 0   # Config C6
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
    rconfig["optimizer"] = 'adam'
    rconfig["initial_learning_rate"] = 27e-6

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
    settings_dict["activation"] = ('softmax', 'sigmoid', 'softmax', 'sigmoid', 'softmax', 'sigmoid')
    settings_dict["optimizer"] = ('adam', 'adam', 'sgd_nesterov', 'sgd_nesterov', 'sgd_momentum', 'sgd_momentum')
    settings_dict["initial_learning_rate"] = (27e-6, 27e-6, 1e-3, 1e-3, 1e-6, 1e-6)

    # Start experiments
    cnn_experiment_procedure(rconfig, settings_dict, experiment_subfolder)
