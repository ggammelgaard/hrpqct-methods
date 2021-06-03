import os
import datetime
import pandas as pd
import numpy as np

from unet.train import train
from unet.predict import predict
from unet.evaluate import evaluate
from gpu_init import startup_limit_gpu


def unet_experiment_procedure(mconfig, settings_dict, experiment_subfolder=None):
    startup_limit_gpu()  # ensure that others can use the machine

    # Convert settings dict to a dataframe
    df = pd.DataFrame(data=settings_dict)

    # Print status
    print("Dataframe for experiments:")
    print(df)
    print("")
    for index, row in df.iterrows():  # iterate through every row of dataframe
        print("Creating model {} of {}".format(index + 1, df.shape[0]))

        # Ensure that values have datatype 'int' and not 'numpy.int64' and similar
        row_dict = dict(row)
        for key, value in row_dict.items():
            if type(value).__module__ == np.__name__:
                row_dict[key] = value.item()
            # print("value: {}, type(value): {}".format(row_dict[key], type(row_dict[key])))

        print("Model settings:", row_dict)
        mconfig.update(row_dict)  # update mconfig with settings for this iteration

        # Decide model subfolder
        if experiment_subfolder:
            unet_subfolder = os.path.join(experiment_subfolder, datetime.datetime.now().strftime("%y%m%d_%H%M"))
        else:
            unet_subfolder = datetime.datetime.now().strftime("%y%m%d_%H%M")

        # Train model
        start_time = datetime.datetime.now()
        train(mconfig, unet_subfolder)
        stop_time = datetime.datetime.now()
        delta_time = (stop_time - start_time)
        print("Training finished for experiment {}".format(index + 1))
        print("Start time: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
        print("Stop time: {}".format(stop_time.strftime("%Y-%m-%d %H:%M:%S")))
        print("Time elapsed: {}".format(delta_time - datetime.timedelta(microseconds=delta_time.microseconds)))
        print("unet_subfolder: {}".format(unet_subfolder))
        print("")

        # Predict model
        predict(mconfig, unet_subfolder)
        # Evaluate model
        evaluate(mconfig, unet_subfolder)

        # Print status
        print("Finished model {} of {}".format(index + 1, df.shape[0]))
        print("unet_subfolder: {}".format(unet_subfolder))
        print("")
        print("")
