import sys
import os
import glob
import datetime
import json

# project imports
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from core.data import write_data_to_file, open_data_file
from core.generator import get_training_and_validation_generators
from core.model import isensee2017_model
from unet.unet_config import get_kfold_configuration, get_unet_configuration
from core.training import load_old_model, train_model
from core.metrics import update_global_weights

def get_model_memory_usage(batch_size, model):
    """ Based on https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model"""
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def main():
    # variables
    kconfig = get_kfold_configuration("")
    mconfig = get_unet_configuration(kconfig)

    # Change settings manually
    # mconfig["input_shape"] = (1,16,16,16)

    # instantiate new model
    update_global_weights(mconfig)
    model_name = 'estimator_model'
    model = isensee2017_model(input_shape=mconfig["input_shape"], n_labels=mconfig["n_labels"],
                              initial_learning_rate=mconfig["initial_learning_rate"],
                              n_base_filters=mconfig["n_base_filters"],
                              model_name=model_name)

    gbytes = get_model_memory_usage(mconfig["batch_size"], model)
    print(gbytes)

    os.remove(model_name)  # The model function auto saves this dummy model.


if __name__ == "__main__":
    main()
