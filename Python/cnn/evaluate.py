import sys
import numpy as np
import nibabel as nib
import os
import glob
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects

from cnn.cnn_config import cnn_model_root_path, load_rconfig
from unet.evaluate import loss_graph
from core.utils.utils import resize


def evaluate(rconfig, cnn_subfolder):
    # Define paths
    train_subfolder = os.path.join(cnn_model_root_path, cnn_subfolder, 'train')
    predict_subfolder = os.path.join(cnn_model_root_path, cnn_subfolder, 'predict')
    output_folder = os.path.join(cnn_model_root_path, cnn_subfolder, 'evaluate')
    try:
        os.makedirs(output_folder)
    except:
        pass
    print("Output folder:", output_folder)

    # Find all folders inside prediction folder
    glob_list = glob.glob(os.path.join(predict_subfolder, '*'))
    folders = list()
    for path in glob_list:
        if os.path.isdir(path):
            folders.append(path)
    folders.sort()
    print("Finding scores for {} cases".format(len(folders)))

    # Calculate correct score
    n_correct = 0
    n_measures = 0
    for folder in folders:
        with open(os.path.join(folder, 'prediction.json')) as json_file:
            results = json.load(json_file)
            if results["prediction"] == results["truth"]:
                n_correct += 1
        n_measures += 1
    # Accuracy score
    acc = n_correct / n_measures
    print("Accuracy score: {:.4f}".format(acc))
    with open(os.path.join(output_folder, "prediction_score.txt"), "w") as text_file:
        text_file.write("Accuracy score: {:.4f}\n".format(acc))
        text_file.write("n_measures: {:.4f}\n".format(n_measures))
        text_file.write("n_correct: {:.4f}\n".format(n_correct))

    # Make loss graph
    print('Creating loss graph')
    train_log = os.path.join(train_subfolder, 'training.log')
    loss_graph(train_log, output_folder)


def main(cnn_subfolder):
    # Get configuration
    rconfig = load_rconfig(cnn_subfolder)

    # Train model
    evaluate(rconfig, cnn_subfolder)


if __name__ == "__main__":
    cnn_subfolder = "cnn_exp_01"
    main(cnn_subfolder)
