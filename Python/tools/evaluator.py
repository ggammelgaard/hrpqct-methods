import os
import sys
import glob
import cc3d
import pandas as pd
import numpy as np
import itertools

sys.path.insert(0, os.path.abspath('../'))
from core.metrics import numpy_soft_dice_coeff
from core.generator import get_onehot_labelmap, get_unique_patient_finger_list
from contextlib import redirect_stdout
from cnn.convert import get_cc_map


def eval_settings(prediction_type):
    if prediction_type == 'hnh':
        labels = (0, 4)
        hnh = True
        rcnn = False
        header = ["Healthy", "Non-healthy"]
    elif prediction_type == 'fm':
        labels = (0, 3)
        hnh = False
        rcnn = False
        header = ["Background", "Erosion"]
    elif prediction_type in ['tuber', 'resnet', 'vggnet']:
        labels = (0, 2, 3,)
        hnh = False
        rcnn = True
        header = ["Background", "Cyst", "Erosion"]
    else:
        labels = (0, 1, 2, 3)
        hnh = False
        rcnn = False
        header = ["Background", "Bone", "Cyst", "Erosion"]

    return labels, header, hnh, rcnn


def eval_dice(labels, truth, truth_onehot, prediction_onehot):
    # Find dice_scores
    dice = numpy_soft_dice_coeff(truth_onehot, prediction_onehot)
    # Ensure that empty classes are not counted towards scores
    ignored_classes = np.setxor1d(np.unique(truth), labels)
    for label in ignored_classes:
        if (len(labels) == 1) and (label in labels):
            dice[1] = None
        else:
            dice[np.where(labels == label)] = None

    return dice


def eval_quantification(quant_scores, y_true, y_pred, axis=(-3, -2, -1)):
    # Calc positives
    true_positive = np.sum(y_true * y_pred, axis=axis)
    false_negative = np.sum(y_true, axis=axis) - true_positive
    # Find inverse
    y_true_inv = np.logical_not(y_true).astype(np.float64)
    y_pred_inv = np.logical_not(y_pred).astype(np.float64)
    # Calc negatives
    true_negative = np.sum(y_true_inv * y_pred_inv, axis=axis)
    false_positive = np.sum(y_true_inv, axis=axis) - true_negative
    # Update quant_scores
    quant_scores[0] += true_positive
    quant_scores[1] += false_negative
    quant_scores[2] += true_negative
    quant_scores[3] += false_positive


def eval_detection(y_true, y_pred):
    """
    lav en liste over alle CC indexes
    for hver true-CC, se om nogen af pred-CC har samme indexes
    TP hvis ja
    FN hvis nej
    FP = for hver pred-CC, se om der er ingen af true-CC der har samme indexes
    """
    n_skips = detection_n_skips(y_true.shape[0])
    if n_skips == -1:
        print("ERROR! UNKNOWN NUMBER OF SKIPS, DETECTION SCORE NOT EVALUATED.")
        return 0, 0, 0

    # For detection
    d_true_positive = np.zeros(y_true.shape[0] - n_skips)  # we do not test cc on background
    d_false_negative = np.zeros(y_true.shape[0] - n_skips)
    d_false_positive = np.zeros(y_true.shape[0] - n_skips)
    # For detection segmentation
    detect_dice_scores = [[] for _ in range(y_true.shape[0] - n_skips)]
    # s_true_positive = np.zeros(y_true.shape[0] - n_skips)
    # s_true_negative = np.zeros(y_true.shape[0] - n_skips)
    # s_false_negative = np.zeros(y_true.shape[0] - n_skips)
    # s_false_positive = np.zeros(y_true.shape[0] - n_skips)
    for i in range(n_skips, y_true.shape[0]):
        true_cc = get_cc_map(y_true[i])
        pred_cc = get_cc_map(y_pred[i])

        # find TP and FN
        for tlabel, tcc in cc3d.each(true_cc, binary=True, in_place=True):
            intersect = pred_cc[tcc > 0]
            if np.count_nonzero(intersect):
                d_true_positive[i - n_skips] += 1

                ## Find detected segmentation accuracy ##
                intersecting_regions = np.zeros_like(tcc)
                # Find all regions that overlaps with the truth
                for plabel, pcc in cc3d.each(pred_cc, binary=True, in_place=True):
                    tmp_intersect = pcc[tcc > 0]
                    if np.count_nonzero(tmp_intersect):
                        intersecting_regions += pcc
                # Calculate detected dice score
                # print(np.count_nonzero(intersecting_regions))
                tmp_quant_scores = [0, 0, 0, 0]
                eval_quantification(tmp_quant_scores, tcc, intersecting_regions)
                s_true_positive = tmp_quant_scores[0]
                s_false_negative = tmp_quant_scores[1]
                s_false_positive = tmp_quant_scores[3]
                dice = (2 * s_true_positive) / (2 * s_true_positive + s_false_positive + s_false_negative)
                detect_dice_scores[i - n_skips].append(dice)
            else:
                d_false_negative[i - n_skips] += 1
        # find FP
        for label, cc in cc3d.each(pred_cc, binary=True, in_place=True):
            intersect = true_cc[cc > 0]
            if np.count_nonzero(intersect) == 0:
                d_false_positive[i - n_skips] += 1

    # # Update detection_scores
    # detection_scores[0] += true_positive
    # detection_scores[1] += false_negative
    # detection_scores[2] += false_positive
    return d_true_positive, d_false_negative, d_false_positive, detect_dice_scores


def eval_reliability(detection_dict, subject_ids):
    # Find number classes that have been detected:
    n_classes = len(next(iter(detection_dict.values()))[0])
    # Make list of fingers with more than one scan
    unique_list = get_unique_patient_finger_list(None, subject_ids)
    consecutive_list = [x for x in unique_list if len(x) > 1]
    # Find erosion count increase for every pair
    increase_list = list()  # [0] = first, [1] = second, [2] = increase
    for finger_scans in consecutive_list:
        for i in range(1, len(finger_scans)):
            first_subject = subject_ids[finger_scans[i - 1]]
            second_subject = subject_ids[finger_scans[i]]
            increment_tp = detection_dict[first_subject][0] - detection_dict[second_subject][0]
            increment_fp = detection_dict[first_subject][1] - detection_dict[second_subject][1]
            increment_tot = increment_tp  # + increment_fp
            increase_list.append([first_subject, second_subject, increment_tot])
    # Sort in positive and negative
    increase_list = np.array(increase_list)
    zero_or_positive = list()
    negative = list()
    n_positive = list()
    n_negative = list()
    var_positive = list()
    var_negative = list()
    for i in range(n_classes):
        zero_or_positive.append(increase_list[np.stack(increase_list[:, 2])[:, i] >= 0])
        negative.append(increase_list[np.stack(increase_list[:, 2])[:, i] < 0])
        # Count N_positive and N_negative
        n_positive.append(len(zero_or_positive[i]))
        n_negative.append(len(negative[i]))
        # Compute variance of N_positive and N_negative
        try:
            var_positive.append(np.var(np.stack(zero_or_positive[i][:, 2])[:, i]))
        except IndexError:
            var_positive.append(0.0)
        try:
            var_negative.append(np.var(np.stack(negative[i][:, 2])[:, i]))
        except IndexError:
            var_negative.append(0.0)
    return (n_positive, n_negative, var_positive, var_negative)


def detection_n_skips(n_labels):
    if n_labels == 4:  # skip background and bone
        n_skips = 2
    elif n_labels == 3:  # skip background
        n_skips = 1
    elif n_labels == 2:  # skip background
        n_skips = 1
    else:
        n_skips = -1
    return n_skips


def save_dice(dice_coeffs, header, subject_ids, output_path, prediction_type):
    # Expand header
    new_header = header.copy()
    for i in range(len(header)):
        new_header[i] = 'DICE_' + new_header[i]

    # Save dice coefficients
    dice_dataframe = pd.DataFrame.from_records(dice_coeffs, columns=header, index=subject_ids)
    dice_score_path = os.path.join(output_path, prediction_type + "_dice_scores.csv")
    dice_dataframe.to_csv(dice_score_path, float_format='%.4f', na_rep='nan')

    # Save dice summary
    summary_path = os.path.join(output_path, prediction_type + '_dice_summary.txt')
    pd.options.display.float_format = '{:,.4f}'.format
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            # Print out median and max values of all classes.
            print("Ncols: {}".format(dice_dataframe.shape[0]))
            print("Max scores:")
            max_score = dice_dataframe.max(axis=0).rename('max_score')
            max_index = dice_dataframe.idxmax(axis=0).rename('max_index')
            print(pd.concat([max_score, max_index], axis=1))
            print()
            print("Median scores:")
            median_score = dice_dataframe.median(axis=0).rename('median_score')
            print(median_score)
            print()
            print("Average of individual scores:")
            average_score = dice_dataframe.mean(axis=0).rename('average_score')
            print(average_score)

            print()
            print("Count of non-NAN values:")
            print(dice_dataframe.count())
            print()
            print("Count of non-zero values:")
            print(dice_dataframe.fillna(0).astype(bool).sum(axis=0))


def save_detection(detection_scores, header, output_path, prediction_type):
    n_skips = detection_n_skips(len(header))
    if n_skips == -1:
        print("ERROR! UNKNOWN NUMBER OF SKIPS, DETECTION SCORE NOT EVALUATED.")
        return detection_scores

    # Change to names we understand
    true_positive = detection_scores[0]
    false_negative = detection_scores[1]
    false_positive = detection_scores[2]
    # Calculate stats
    sensitivity = true_positive / (true_positive + false_negative)
    ppv = true_positive / (true_positive + false_positive)

    # Expand header
    n_stats = 5
    new_header = [''] * (len(header) - n_skips) * n_stats  # we do not save stats for background
    for i in range(len(header) - n_skips):
        new_header[i + 0 * (len(header) - n_skips)] = 'TP_' + header[i + n_skips]
        new_header[i + 1 * (len(header) - n_skips)] = 'FN_' + header[i + n_skips]
        new_header[i + 2 * (len(header) - n_skips)] = 'FP_' + header[i + n_skips]
        new_header[i + 3 * (len(header) - n_skips)] = 'TPR_' + header[i + n_skips]
        new_header[i + 4 * (len(header) - n_skips)] = 'PPV_' + header[i + n_skips]

    # Save values
    dataframe = pd.DataFrame.from_records(
        np.concatenate([true_positive, false_negative, false_positive, sensitivity, ppv]).reshape([len(new_header), 1]),
        index=new_header)
    dataframe_path = os.path.join(output_path, prediction_type + "_detection_scores.csv")
    dataframe.to_csv(dataframe_path, float_format='%.4f', na_rep='nan', header=False)


def save_quantification(quant_scores, header, output_path, prediction_type):
    # Change to names we understand
    true_positive = quant_scores[0]
    false_negative = quant_scores[1]
    true_negative = quant_scores[2]
    false_positive = quant_scores[3]
    # Calculate stats
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    ppv = true_positive / (true_positive + false_positive)
    dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

    # Expand header
    n_stats = 4
    new_header = [''] * len(header) * n_stats
    for i in range(len(header)):
        new_header[i] = 'TPR_' + header[i]
        new_header[i + 1 * len(header)] = 'TNR_' + header[i]
        new_header[i + 2 * len(header)] = 'PPV_' + header[i]
        new_header[i + 3 * len(header)] = 'DICE_' + header[i]
    # Save values
    dataframe = pd.DataFrame.from_records(
        np.concatenate([sensitivity, specificity, ppv, dice]).reshape([len(new_header), 1]),
        index=new_header)
    dataframe_path = os.path.join(output_path, prediction_type + "_quantification_scores.csv")
    dataframe.to_csv(dataframe_path, float_format='%.4f', na_rep='nan', header=False)


def save_reliability(reliability_stats, header, output_path, prediction_type):
    n_skips = detection_n_skips(len(header))
    if n_skips == -1:
        print("ERROR! UNKNOWN NUMBER OF SKIPS, RELIABILITY SCORE NOT EVALUATED.")
        return reliability_stats

    # Change to names we understand
    n_positive = np.array(reliability_stats[0])
    n_negative = np.array(reliability_stats[1])
    var_positive = np.array(reliability_stats[2])
    var_negative = np.array(reliability_stats[3])
    # Calculate stats
    pos_ratio = n_positive / (n_positive + n_negative)

    # Expand header
    n_stats = 5
    new_header = [''] * (len(header) - n_skips) * n_stats  # we do not save stats for background
    for i in range(len(header) - n_skips):
        new_header[i + 0 * (len(header) - n_skips)] = 'n_positive' + '_' + header[i + n_skips]
        new_header[i + 1 * (len(header) - n_skips)] = 'n_negative' + '_' + header[i + n_skips]
        new_header[i + 2 * (len(header) - n_skips)] = 'var_positive' + '_' + header[i + n_skips]
        new_header[i + 3 * (len(header) - n_skips)] = 'var_negative' + '_' + header[i + n_skips]
        new_header[i + 4 * (len(header) - n_skips)] = 'pos_ratio' + '_' + header[i + n_skips]

    # Save values
    dataframe = pd.DataFrame.from_records(
        np.concatenate([n_positive, n_negative, var_positive, var_negative, pos_ratio]).reshape([len(new_header), 1]),
        index=new_header)
    dataframe_path = os.path.join(output_path, prediction_type + "_reliability_scores.csv")
    dataframe.to_csv(dataframe_path, float_format='%.4f', na_rep='nan', header=False)


def save_detected_seg(detected_dice_scores, header, output_path, prediction_type):
    n_skips = detection_n_skips(len(header))
    if n_skips == -1:
        print("ERROR! UNKNOWN NUMBER OF SKIPS, DETECTION SCORE NOT EVALUATED.")
        return detected_dice_scores

    new_header = header.copy()
    new_header = new_header[n_skips:]
    for i in range(len(new_header)):
        new_header[i] = 'DICE_' + new_header[i]

    # Save dice coefficients
    zip_tuple = (_ for _ in itertools.zip_longest(*detected_dice_scores))
    dice_dataframe = pd.DataFrame.from_records(zip_tuple, columns=new_header)
    dataframe_path = os.path.join(output_path, prediction_type + "_dq_scores.csv")
    dice_dataframe.to_csv(dataframe_path, float_format='%.4f', na_rep='nan')

    # Save dice summary
    summary_path = os.path.join(output_path, prediction_type + '_dq_summary.txt')
    pd.options.display.float_format = '{:,.4f}'.format
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            # Print out median and max values of all classes.
            print("Ncols: {}".format(dice_dataframe.shape[0]))
            print("Max scores:")
            max_score = dice_dataframe.max(axis=0).rename('max_score')
            max_index = dice_dataframe.idxmax(axis=0).rename('max_index')
            print(pd.concat([max_score, max_index], axis=1))
            print()
            print("Min scores:")
            min_score = dice_dataframe.min(axis=0).rename('min_score')
            min_index = dice_dataframe.idxmin(axis=0).rename('min_index')
            print(pd.concat([min_score, min_index], axis=1))
            print()
            print("Median scores:")
            median_score = dice_dataframe.median(axis=0).rename('median_score')
            print(median_score)
            print()
            print("Average of individual scores:")
            average_score = dice_dataframe.mean(axis=0).rename('average_score')
            print(average_score)

            print()
            print("Count of non-NAN values:")
            print(dice_dataframe.count())
            print()
            print("Count of non-zero values:")
            print(dice_dataframe.fillna(0).astype(bool).sum(axis=0))


def evaluator(prediction_type, prediction_path, output_path, resize_shape, quantification, detection):
    # Find all folders inside prediction folder
    glob_list = glob.glob(os.path.join(prediction_path, '*'))
    folders = list()
    for path in glob_list:
        if os.path.isdir(path):
            folders.append(path)
    folders.sort()
    print("Evaluating scores for {} cases".format(len(folders)))

    # Decide on run settings
    labels, header, hnh, rcnn = eval_settings(prediction_type)

    # Computation loop
    # folders = folders[0:2]  # debug
    counter = 1
    dice_scores = list()
    quantification_scores = [0] * 4  # tp, fn, tn, fp
    detected_dice_scores = list()
    detection_scores = [0] * 3  # tp, fn, fp
    detection_dict = dict()
    subject_ids = list()
    for case_folder in folders:
        # Add subject to ouput list
        subject = os.path.basename(case_folder)
        subject_ids.append(subject)
        # Get truth data
        if hnh == True:  # healthy / non-healthy
            truth_path = glob.glob(os.path.join(case_folder, '*_hnh.nii.gz'))[0]
        else:
            truth_path = glob.glob(os.path.join(case_folder, "*_truth.nii.gz"))[0]
        truth_onehot, truth = get_onehot_labelmap(truth_path, labels, resize_shape)
        # Get prediction data
        if rcnn == True:  # rcnn predictions
            prediction_path = glob.glob(os.path.join(case_folder, '*_' + prediction_type + '.nii.gz'))[0]
        else:
            prediction_path = glob.glob(os.path.join(case_folder, '*_prediction.nii.gz'))[0]
        prediction_onehot, _ = get_onehot_labelmap(prediction_path, labels, resize_shape)

        # Evaluate
        if quantification:
            dice_scores.append(eval_dice(labels, truth, truth_onehot, prediction_onehot))
            eval_quantification(quantification_scores, truth_onehot, prediction_onehot)
        if detection:
            dtp, dfn, dfp, detect_ds = eval_detection(truth_onehot, prediction_onehot)
            detection_scores[0] += dtp
            detection_scores[1] += dfn
            detection_scores[2] += dfp
            detection_dict[subject] = [dtp, dfn, dfp]
            detected_dice_scores.append(detect_ds)

        # Status
        print("Index: {}, Subject: {}".format(counter, subject))
        counter += 1

    # Post evaluate reliability
    reliability_stats = eval_reliability(detection_dict, subject_ids)
    # Format detected_dice_scores
    clean_detected_dice_scores = list()
    for i in range(len(detected_dice_scores[0])):
        clean_detected_dice_scores.append(np.concatenate(np.array(detected_dice_scores).T[i].flatten()))

    # Save files
    if quantification:
        save_dice(dice_scores, header, subject_ids, output_path, prediction_type)
        save_quantification(quantification_scores, header, output_path, prediction_type)
    if detection:
        save_detection(detection_scores, header, output_path, prediction_type)
        save_reliability(reliability_stats, header, output_path, prediction_type)
        save_detected_seg(clean_detected_dice_scores, header, output_path, prediction_type)


if __name__ == "__main__":
    # Define paths
    # prediction_path = r'C:\school\RnD\trash'
    # prediction_path = r'C:\school\RnD\server_data\210502_1026_exp_04_HER\config_2_predictions'
    prediction_path = r'C:\school\RnD\server_data\210511_1345_rcnn_B3_C15\combined_predictions'
    output_path = prediction_path
    # output_path = r'C:\school\RnD'

    prediction_type = 'tuber'  # Prediction type
    resize_shape = (128, 128, 128)  # Resize to this shape
    detection = True  # whether or not to include detection
    quantification = True  # whether or not to include quantification and dice

    print("prediction_path:", prediction_path)
    print("output_path:", output_path)
    print("prediction_type:", prediction_type)
    print("data_shape:", resize_shape)
    print("quantification:", quantification)
    print("detection:", detection)

    # Run stats script
    evaluator(prediction_type, prediction_path, output_path, resize_shape, quantification, detection)
