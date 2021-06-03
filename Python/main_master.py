#!/usr/bin/env python3
import os
import sys
import datetime
from gpu_init import startup_limit_gpu


def get_main(task):
    # Decice on which main should be run
    if task == 'unet_kfold':
        package = "unet.kfold"
        required_folders = ["kfold_subfolder"]
    elif task == 'unet_train':
        package = "unet.train"
        required_folders = ["kfold_subfolder", "unet_subfolder"]
    elif task == 'unet_predict':
        package = "unet.predict"
        required_folders = ["unet_subfolder"]
    elif task == 'unet_evaluate':
        package = "unet.evaluate"
        required_folders = ["unet_subfolder"]
    elif task == 'cnn_convert':
        package = "cnn.convert"
        required_folders = ["kfold_subfolder"]
    elif task == 'cnn_train':
        package = "cnn.train"
        required_folders = ["kfold_subfolder", "cnn_subfolder"]
    elif task == 'cnn_predict':
        package = "cnn.predict"
        required_folders = ["cnn_subfolder"]
    elif task == 'cnn_predict':
        package = "cnn.predict"
        required_folders = ["cnn_subfolder"]
    elif task == 'cnn_evaluate':
        package = "cnn.evaluate"
        required_folders = ["cnn_subfolder"]
    elif task == 'rcnn_predict':
        package = "tools.rcnn_predictor"
        required_folders = ["unet_subfolder", "cnn_subfolder"]
    else:
        print("ERROR: TASK NOT FOUND!")
        sys.exit()

    # Fetch the main
    sys.path.insert(0, os.path.abspath(''))
    main = getattr(__import__(package, fromlist=['main']), 'main')
    return main, required_folders


if __name__ == "__main__":
    # Get task
    try:
        task = sys.argv[1]
    except:
        print("ABORTED. NO ARGUMENTS GIVEN!")
        sys.exit()

    # Find corresponding main
    main, required_folders = get_main(task)

    # Get input_arguments
    input_arguments = list()
    for i in range(len(required_folders)):
        try:
            input_arguments.append(sys.argv[i + 2])
        except IndexError:
            print("ABORTED. {} ARGUMENTS NEEDED: {}".format(len(required_folders), required_folders))
            sys.exit()

    # Start time
    start_time = datetime.datetime.now()

    # Start program
    startup_limit_gpu()
    main(*input_arguments)

    # Stop time
    stop_time = datetime.datetime.now()
    delta_time = (stop_time - start_time)
    # Indicate program has finished
    print("")
    print("{} finished.".format(os.path.basename(__file__)))
    print("Start time: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    print("Stop time: {}".format(stop_time.strftime("%Y-%m-%d %H:%M:%S")))
    print("Time elapsed: {}".format(delta_time - datetime.timedelta(microseconds=delta_time.microseconds)))
    print()
    for i in range(len(required_folders)):
        print("{}: {}".format(required_folders[i], input_arguments[i]))
    print("")
