import os
import glob
import nibabel as nib
import numpy as np
import cc3d
from unet.unet_config import load_mconfig, unet_model_root_path
from cnn.cnn_config import load_rconfig, cnn_model_root_path
from cnn.predict import cnn_model_prediction
from core.generator import get_onehot_labelmap
from cnn.convert import get_cc_map, bbox2_3d, increase_boundary
from core.training import load_old_model
from core.utils.utils import resize


def predict(mconfig, rconfig, unet_subfolder, cnn_subfolder):
    # Define paths
    cnn_train_path = os.path.join(cnn_model_root_path, cnn_subfolder, 'train')
    cnn_model_path = os.path.join(cnn_train_path, rconfig["model_name"])
    unet_predict_path = os.path.join(unet_model_root_path, unet_subfolder, 'predict')

    # Find all folders inside prediction folder
    glob_list = glob.glob(os.path.join(unet_predict_path, 'P*'))
    folders = list()
    for path in glob_list:
        if os.path.isdir(path):
            folders.append(path)
    folders.sort()

    # Load cnn model
    cnn_model = load_old_model(cnn_model_path)

    # Make predictions
    print("Computing statistical scores for {} cases".format(len(folders)))
    labels = (4,)
    i_status = 1
    for case_folder in folders:
        subject = os.path.basename(case_folder)
        print("Predicting file no. {}: {}".format(i_status, subject))
        # Get prediction data
        unet_prediction_path = glob.glob(os.path.join(case_folder, '*_prediction.nii.gz'))[0]
        unet_prediction_onehot, _ = get_onehot_labelmap(unet_prediction_path, labels)
        # Get real data
        data_path = glob.glob(os.path.join(case_folder, '*_data.nii.gz'))[0]
        data_image = nib.load(data_path)
        # Create empty tensor for new prediction
        rcnn_prediction = np.zeros_like(unet_prediction_onehot[1])

        # Crop out CC's (don't remove small things)
        true_cc = get_cc_map(unet_prediction_onehot[1], connectivity=26, voxel_threshold=0)
        binc = 6  # border increase
        for label, cc in cc3d.each(true_cc, binary=True, in_place=True):
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3d(cc)  # find bb
            rmin, rmax, cmin, cmax, zmin, zmax = increase_boundary(binc, unet_prediction_onehot[1].shape, rmin, rmax,
                                                                   cmin,
                                                                   cmax, zmin, zmax)
            # Crop image
            data_cc = data_image.slicer[rmin:rmax, cmin:cmax, zmin:zmax]
            # Resize crop
            data_rez = resize(data_cc, rconfig["cc_shape"], interpolation="linear")

            # Make prediction
            answer, multiplier = cnn_model_prediction(cnn_model, data_rez.get_data()[np.newaxis][np.newaxis])

            # Update output map
            rcnn_prediction[cc > 0] = multiplier

        # Save in same folder
        rcnn_image = nib.Nifti1Image(rcnn_prediction, data_image.affine)
        rcnn_image.to_filename(os.path.join(case_folder, subject + "_" + rconfig["model_type"] + ".nii.gz"))

        i_status += 1


def main(unet_subfolder, cnn_subfolder):
    # Get configurations
    rconfig = load_rconfig(cnn_subfolder)
    mconfig = load_mconfig(unet_subfolder)

    # Train model
    predict(mconfig, rconfig, unet_subfolder, cnn_subfolder)


if __name__ == "__main__":
    unet_subfolder = "210301_1516"
    cnn_subfolder = "cnn_exp_01"
    main(unet_subfolder, cnn_subfolder)
