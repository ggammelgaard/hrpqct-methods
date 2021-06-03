#!/usr/bin/env python3
""" EXP_14 INDEX 0-4 HNH Config B2
45	Softmax		DL		w_Frequency   IDX 0
46	Softmax		DL		w_Frequency   IDX 1
47	Softmax		DL		w_Frequency   IDX 2
48	Softmax		DL		w_Frequency   IDX 3
49	Softmax		DL		w_Frequency   IDX 4
w_Frequency = (1.00, 2.87)
"""
import os
import sys
import datetime
import pandas as pd

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(sys.argv[0])))
    from unet.unet_config import get_unet_configuration, load_kconfig
    from unet_experiments.experiment_protocol import unet_experiment_procedure

    # Ensure that variables are entered
    try:
        kfold_subfolder = sys.argv[1]
        experiment_subfolder = sys.argv[2]
    except:
        print("ABORTED. 2 ARGUMENTS NEEDED: 'kfold_subfolder' and 'experiment_subfolder'")
        sys.exit()

    # Get configuration
    kconfig = load_kconfig(kfold_subfolder)
    mconfig = get_unet_configuration(kconfig)

    # Make test base modifications
    mconfig["non_healthy"] = True  # Override with healthy vs. non_healthy labeling
    mconfig["label_weights"] = (1.00, 2.87)
    mconfig["labels"] = (0, 1)
    mconfig["n_labels"] = len(mconfig["labels"])
    mconfig["n_epochs"] = 200  # cutoff the training after this many epochs
    mconfig["kfold_index"] = 0
    mconfig["activation"] = 'softmax'  # 'sigmoid' or 'softmax'
    mconfig["loss"] = 'dl'  # 'dl', 'ce', 'gdl'
    mconfig["flip"] = False  # augments the data by randomly flipping an axis during
    mconfig["distort"] = None  # True  # switch to None if you want no distortion
    mconfig["augment"] = mconfig["flip"] or mconfig["distort"]
    mconfig["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
    mconfig["initial_learning_rate"] = 5e-4  # consider using 0.0002 as they do in wolny

    # Make individual settings
    settings_dict = dict()
    settings_dict["kfold_index"] = (0, 1, 2, 3, 4)

    # Start experiments
    unet_experiment_procedure(mconfig, settings_dict, experiment_subfolder)
