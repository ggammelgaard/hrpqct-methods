""" Finds meta information and makes plots from dice score csv file.
"""
import os
import pandas as pd
from contextlib import redirect_stdout

from unet.evaluate import validation_boxplot, loss_graph


def get_median_index(d):
    ranks = d.rank(pct=True)
    close_to_median = abs(ranks - 0.5)
    return close_to_median.idxmin()


def dice_score_evaluator(dice_score_path, output_path, training_log_path=None, figsize=([6.4, 4.8])):
    # Read csv
    data = pd.read_csv(dice_score_path, index_col=0)

    # Make boxplot
    validation_boxplot(data, output_path, figsize)

    # # Make loss graph
    if training_log_path:
        loss_graph(training_log_path, output_path, figsize)

    # Create log file
    logpath = os.path.join(output_path, 'dice_log.txt')
    with open(logpath, 'w') as f:
        with redirect_stdout(f):
            # Print out median and max values of all classes.
            print("Ncols: {}".format(data.shape[0]))
            print("Max DICE scores:")
            max_score = data.max(axis=0).rename('max_score')
            max_index = data.idxmax(axis=0).rename('max_index')
            print(pd.concat([max_score, max_index], axis=1))
            print()
            print("Median DICE scores:")
            median_score = data.median(axis=0).rename('median_score')
            print(median_score)
            print()
            print("Average DICE scores:")
            average_score = data.mean(axis=0).rename('average_score')
            print(average_score)

            print()
            print("Count of non-NAN values:")
            print(data.count())
            print()
            print("Count of non-zero values:")
            print(data.fillna(0).astype(bool).sum(axis=0))


if __name__ == '__main__':
    # Paths
    output_path = r'C:\school\RnD'
    dice_score_path = r'C:\school\RnD\server\210312_1730_exp_08\combined_dice_scores.csv'
    # training_log_path = r'C:\school\RnD\server\210303_0851\train\training.log'
    training_log_path = None

    # Run script
    dice_score_evaluator(dice_score_path, output_path, training_log_path, figsize=([6.4, 4.8]))
