import os
import sys
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


# Globally change font to LaTeX font
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False

def dq_boxplot(dice_coefficient_dataframe, output_folder, figsize=[6.4, 3.5], outname='evaluation_boxplot'):
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
                f'{value:.3f}',
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
        boxprops=dict(facecolor=(0.69, 0.48, 0.40, 0.8)),
        widths=0.4,
        # showmeans=True,
    )

    # # fill with colors
    # if dice_coefficient_dataframe.shape[1] == 2:  # healthy / non-healthy scenario
    #     # colors = ['#5A5A5A', '#6FB8D2']
    #     colors = [(0.35, 0.35, 0.35, 0.8), (0.44, 0.72, 0.82, 0.8)]
    # else:
    #     # colors = ['#5A5A5A', '#80AE80', '#F1D691', '#B17A65']
    #     colors = [(0.35, 0.35, 0.35, 0.8), (0.5, 0.68, 0.5, 0.8), (0.95, 0.84, 0.57, 0.8), (0.69, 0.48, 0.40, 0.8)]
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

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


if __name__ == '__main__':
    output_path = r'C:\school\RnD'

    # Get dataframe paths
    # data_path_1 = r'C:\school\RnD\server_data\210502_1026_exp_04_HER\config_2_predictions'
    data_path_1 = r'C:\school\RnD\server_data\210523_multi\fm_dq_scores.csv'
    data_path_2 = r'C:\school\RnD\server_data\210502_1026_exp_04_HER\config_2_predictions\ml_dq_scores.csv'
    data_path_3 = r'C:\school\RnD\server_data\210511_1345_rcnn_B3_C15\combined_predictions\tuber_dq_scores.csv'

    # Load dataframes
    data_1 = pd.read_csv(data_path_1, index_col=0)
    data_2 = pd.read_csv(data_path_2, index_col=0)
    data_3 = pd.read_csv(data_path_3, index_col=0)

    # Combine frames
    combined_df = pd.concat([data_1.DICE_Erosion, data_2.DICE_Erosion, data_3.DICE_Erosion], axis=1)
    combined_df.columns = ['Hybrid Segmentation', 'U-Net based CNN', 'Region based CNN']
    # print(combined_df)

    # Make boxplot
    dq_boxplot(combined_df, output_path, figsize=[6.4, 3.5], outname='dq_boxplot')