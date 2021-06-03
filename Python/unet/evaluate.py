import sys
import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects

from unet.unet_config import unet_model_root_path, load_mconfig
from tools.evaluator import evaluator

# Globally change font to LaTeX font
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False


def validation_boxplot(dice_coefficient_dataframe, output_folder, figsize=[6.4, 3.5], outname='evaluation_boxplot'):
    def add_median_labels(ax):
        lines = ax.get_lines()
        # determine number of lines per box (this varies with/without fliers)
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        # iterate over median lines
        for median in lines[4:len(lines):lines_per_box]:
            # display median value at center of median line
            x, y = (data.mean() for data in median.get_data())
            # choose value depending on horizontal or vertical plot orientation
            value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
            text = ax.text(
                x,
                y,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontweight='bold',
                size=8,
                color='white',
                bbox=dict(facecolor="#7A7A7A", pad=3))
            # create median-colored border around white text for contrast
            # text.set_path_effects([
            #     path_effects.Stroke(linewidth=2, foreground='black'),
            #     path_effects.Normal(),
            # ])

    scores = dict()
    for index, score in enumerate(dice_coefficient_dataframe.columns):
        values = dice_coefficient_dataframe.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    fig, ax = plt.subplots(figsize=figsize)  # figsize=([6.4, 4.8])
    bplot = ax.boxplot(
        list(scores.values()),
        notch=False,  # notch shape
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=list(scores.keys()),  # will be used to label x-ticks
        medianprops=dict(linewidth=1, color='black'),
        whiskerprops=dict(linestyle=(0, (10, 5))),
        # showmeans=True,
    )
    # fill with colors

    if dice_coefficient_dataframe.shape[1] == 2:  # healthy / non-healthy scenario
        # colors = ['#5A5A5A', '#6FB8D2']
        colors = [(0.35, 0.35, 0.35, 0.8), (0.44, 0.72, 0.82, 0.8)]
    else:
        # colors = ['#5A5A5A', '#80AE80', '#F1D691', '#B17A65']
        colors = [(0.35, 0.35, 0.35, 0.8), (0.5, 0.68, 0.5, 0.8), (0.95, 0.84, 0.57, 0.8), (0.69, 0.48, 0.40, 0.8)]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    # adding horizontal grid lines
    ax.yaxis.grid(True, color="#CFCFCF")
    ax.set_ylabel("Dice Coefficient")
    ax.set_ylim(-0.03, 1.03)
    # add median labels
    add_median_labels(ax)
    # Save
    plt.tight_layout()  # ensures that text are not clipped when saving as pdf
    plt.savefig(os.path.join(output_folder, outname + '.pdf'))
    plt.show()
    plt.close()


def loss_graph(training_log, output_folder, figsize=[6.4, 3.5]):
    if os.path.exists(training_log):
        training_df = pd.read_csv(training_log).set_index('epoch')
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)  # figsize=[6.4, 4.8]
        ax.plot(training_df['loss'].values, label='training', linewidth=1.2, color='#377eb8')
        ax.plot(training_df['val_loss'].values, label='validation', linewidth=1.2, color='#e41a1c')
        # ax.set_ylim(0, 1)
        ax.set_xlim(0)
        # Text
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right')
        # Save
        plt.tight_layout()  # ensures that text are not clipped when saving as pdf
        plt.savefig(os.path.join(output_folder, 'loss_graph.pdf'))
        plt.show()
        plt.close()
    else:
        print('Loss graph aborted. No log file found at {}'.format(training_log))


def evaluate(mconfig, unet_subfolder):
    # Define paths
    train_subfolder = os.path.join(unet_model_root_path, unet_subfolder, 'train')
    predict_subfolder = os.path.join(unet_model_root_path, unet_subfolder, 'predict')
    output_folder = os.path.join(unet_model_root_path, unet_subfolder, 'evaluate')
    try:
        os.makedirs(output_folder)
    except:
        pass
    print("Output folder:", output_folder)

    # Deduce model type
    if mconfig['non_healthy']:
        model_type = 'hnh'
    else:
        model_type = 'ml'

    # Run evaluator script
    evaluator(model_type, predict_subfolder, output_folder, resize_shape=None, quantification=True, detection=True)

    # Make boxplot
    glob_list = glob.glob(os.path.join(output_folder, '*dice_scores.csv'))
    for score_path in glob_list:
        suffix = os.path.basename(score_path)[0:-(len('dice_scores.csv'))]
        dice_dataframe = pd.read_csv(score_path, index_col=0)
        print("Creating boxplot for {}".format(suffix))
        validation_boxplot(dice_dataframe, output_folder, outname=(suffix + 'evaluation_boxplot'))

    # Make loss graph
    print('Creating loss graph')
    train_log = os.path.join(train_subfolder, 'training.log')
    loss_graph(train_log, output_folder)


def main(unet_subfolder):
    # Get configuration
    mconfig = load_mconfig(unet_subfolder)

    # Train model
    evaluate(mconfig, unet_subfolder)


if __name__ == "__main__":
    unet_subfolder = '20210217_1213'
    main(unet_subfolder)
