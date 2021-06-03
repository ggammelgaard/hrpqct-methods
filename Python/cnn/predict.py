import os
import datetime
import sys
import json
import numpy as np
import nibabel as nib

from core.data import open_data_file
from core.training import load_old_model
from cnn.cnn_config import cnn_model_root_path, load_rconfig

def cnn_model_prediction(model, test_data):
    prediction = model.predict(test_data)
    print("prediction: {}, argmax: {}".format(prediction, np.argmax(prediction)))
    if np.argmax(prediction) == 1:
        answer = "erosion"
        multiplier = 3
    else:
        answer = "cyst"
        multiplier = 2

    return answer, multiplier

def predict(rconfig, cnn_subfolder):
    # Define paths
    train_subfolder_path = os.path.join(cnn_model_root_path, cnn_subfolder, 'train')
    output_folder = os.path.join(cnn_model_root_path, cnn_subfolder, 'predict')
    try:
        os.makedirs(output_folder)
    except:
        pass

    model_file_path = os.path.join(train_subfolder_path, rconfig["model_name"])

    # open file
    data_path = os.path.join(rconfig['kfold_path'], 'config_{}_test_cc.h5'.format(rconfig["kfold_index"]))
    assert os.path.exists(data_path)
    data_file = open_data_file(data_path)
    index_list = list(range(data_file.root.data.shape[0]))  # list needed for the run_validation_cases function

    # load old model
    model = load_old_model(model_file_path)
    model.summary()

    # validation_indices = validation_indices[0:1]  # shorten amount of files to be predicted (240 in total)
    print("Number of validation files: {}".format(len(index_list)))
    i_status = 1

    for index in index_list:
        subject = data_file.root.subject_ids[index][0].decode('utf-8')
        print("Predicting file no. {}: Index {} - {}".format(i_status, index, subject))

        # Make output folder
        case_directory = os.path.join(output_folder, subject)
        if not os.path.exists(case_directory):
            os.makedirs(case_directory)

        # Save data as nifti image
        affine = data_file.root.affine[index]
        test_data = np.asarray([data_file.root.data[index]])
        image = nib.Nifti1Image(test_data[0, 0], affine)
        image.to_filename(os.path.join(case_directory, "{}_data.nii.gz".format(subject)))

        # Make prediction
        answer, multiplier = cnn_model_prediction(model, test_data)
        # Save prediction as nifti image
        test_truth = nib.Nifti1Image(data_file.root.truth[index][0] * multiplier, affine)
        test_truth.to_filename(os.path.join(case_directory, "{}_prediction.nii.gz".format(subject)))

        # Save as txt
        out_dict = {'prediction': answer, 'truth': data_file.root.label[index][0].decode('utf-8')}
        with open(os.path.join(case_directory, 'prediction.json'), 'w') as outfile:
            json.dump(out_dict, outfile, indent=4, sort_keys=True)

        i_status += 1

    data_file.close()  # close file again


def main(cnn_subfolder):
    # Get configuration
    rconfig = load_rconfig(cnn_subfolder)

    # Train model
    predict(rconfig, cnn_subfolder)


if __name__ == "__main__":
    cnn_subfolder = "cnn_exp_01"
    main(cnn_subfolder)
