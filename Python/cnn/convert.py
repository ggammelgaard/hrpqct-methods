import os
import numpy as np
import nibabel as nib
import json
import cc3d
import glob
import tables
from core.generator import get_non_healthy_class_labels
from core.data import write_data_to_file, open_data_file
from core.utils.utils import resize
from unet.unet_config import kfold_root_path, get_kfold_configuration
from cnn.cnn_config import get_cc_configuration


def create_cc_data_file(out_file, n_channels, n_samples, n_truth_labels, image_shape):
    """ Initializes the hdf5 file and gives pointers for its three arrays
    """
    try:
        os.makedirs(os.path.dirname(out_file))
    except:
        pass

    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, n_truth_labels] + list(image_shape))

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    subject_storage = hdf5_file.create_earray(hdf5_file.root, 'subject_ids', tables.StringAtom(itemsize=16),
                                              shape=(0, 1),
                                              filters=filters, expectedrows=n_samples)
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'label', tables.StringAtom(itemsize=16), shape=(0, 1),
                                            filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage, subject_storage, label_storage


def get_cc_map(truthmap_3d, connectivity=26, voxel_threshold=30):
    """
    Based on https://pypi.org/project/connected-components-3d/
    :param truthmap_3d:
    :param connectivity: only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    :return:
    """
    labels_im, num_labels = cc3d.connected_components(truthmap_3d.astype(np.uint8), connectivity=connectivity,
                                                      return_N=True)
    # print("pre num_labels", num_labels)
    # print("post labels_im", np.unique(labels_im))
    labels_idx = list()
    for label, cc in cc3d.each(labels_im, binary=True, in_place=True):
        if np.count_nonzero(cc) < voxel_threshold:  # if CC is smaller than 30 voxels
            labels_im[cc > 0] = 0  # delete those voxels
        else:
            labels_idx.append(label)  # label idx will be used later

    return labels_im


def bbox2_3d(img):
    """
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    :param img:
    :return:
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax


def increase_boundary(inc, dshape, rmin, rmax, cmin, cmax, zmin, zmax):
    rmin -= inc
    rmax += inc
    cmin -= inc
    cmax += inc
    zmin -= inc
    zmax += inc

    # Ensure they are within boundaries
    rmin = rmin if rmin > 0 else 0
    rmax = rmax if rmax < dshape[0] else dshape[0] - 1
    cmin = cmin if cmin > 0 else 0
    cmax = cmax if cmax < dshape[1] else dshape[1] - 1
    zmin = zmin if zmin > 0 else 0
    zmax = zmax if zmax < dshape[2] else dshape[2] - 1

    return rmin, rmax, cmin, cmax, zmin, zmax


def convert_hfile(input_path, output_shape):
    # Open input
    input_data = open_data_file(input_path)

    # Open output
    input_filename = os.path.basename(os.path.splitext(input_path)[0])
    output_filename = input_filename + '_cc.h5'
    hfile_path = os.path.join(os.path.dirname(input_path), output_filename)
    n_samples = input_data.root.subject_ids.shape[0]  # just an estimate
    n_truth_labels = 1
    hfile, data_storage, truth_storage, affine_storage, subject_storage, label_storage = create_cc_data_file(
        hfile_path,
        n_channels=1,
        n_samples=n_samples,
        n_truth_labels=n_truth_labels,
        image_shape=output_shape)

    # Loop through data
    for i in range(input_data.root.truth.shape[0]):
        # Load tmp truth map (used to know if it is an erosion or cyst
        tmp_truth = input_data.root.truth[i][0]
        # Create HNH labelmap
        binary_truth = get_non_healthy_class_labels(tmp_truth)
        # Create nib image for crop data and affine
        base_img = nib.Nifti1Image(input_data.root.data[i][0], input_data.root.affine[i])
        base_truth = nib.Nifti1Image(binary_truth, input_data.root.affine[i])

        # Find CCs
        true_cc = get_cc_map(binary_truth)

        binc = 6  # border increase
        for label, cc in cc3d.each(true_cc, binary=True, in_place=True):
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3d(cc)  # find bb
            rmin, rmax, cmin, cmax, zmin, zmax = increase_boundary(binc, binary_truth.shape, rmin, rmax, cmin, cmax,
                                                                   zmin, zmax)
            # Crop image
            data_cc = base_img.slicer[rmin:rmax, cmin:cmax, zmin:zmax]
            truth_cc = base_truth.slicer[rmin:rmax, cmin:cmax, zmin:zmax]
            # Decide on class
            labels, counts = np.unique(input_data.root.truth[i][0][rmin:rmax, cmin:cmax, zmin:zmax], return_counts=True)
            count_cyst = counts[labels == 2] if 2 in labels else 0
            count_erosion = counts[labels == 3] if 3 in labels else 0
            if count_erosion > count_cyst:
                cc_class = 'erosion'
            else:
                cc_class = 'cyst'

            # Resize image and truth
            rez_image = resize(data_cc, output_shape, interpolation="linear")
            rez_truth = resize(truth_cc, output_shape, interpolation="nearest")

            # Save data
            data_storage.append(rez_image.get_data()[np.newaxis][np.newaxis])
            truth_storage.append(rez_truth.get_data()[np.newaxis][np.newaxis])
            affine_storage.append(rez_image.affine[np.newaxis])
            subject_storage.append(np.asarray((input_data.root.subject_ids[i],))[np.newaxis])
            label_storage.append(np.asarray((cc_class,))[np.newaxis])

            # # Save image
            # data_cc.to_filename(
            #     os.path.join(output_path,
            #                  "cc_{}_{}_{}.nii.gz".format(input_data.root.subject_ids[i].decode('utf-8'), label,
            #                                              cc_class)))

    # Close data file
    input_data.close()
    hfile.close()


def convert_kfolds(cconfig, foldername):
    # Find the k h5 files
    search = os.path.join(kfold_root_path, foldername, 'config_*')
    glob_list = set(glob.glob(search)) - set(glob.glob(search + 'cc.h5'))

    n_files = len(glob_list)
    print("len(glob_list):", n_files)

    counter = 1
    for hfile_path in glob_list:
        print("Converting .h5 file {} of {}".format(counter, n_files))
        convert_hfile(hfile_path, cconfig["cc_shape"])

        counter += 1


def main(foldername):
    # Get configuration
    kconfig = get_kfold_configuration(foldername)
    cconfig = get_cc_configuration(kconfig)

    # Create data
    convert_kfolds(cconfig, foldername)

    # Save configuration
    output_path = os.path.join(kfold_root_path, foldername)
    with open(os.path.join(output_path, 'cconfig.json'), 'w') as outfile:
        json.dump(kconfig, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    foldername = "210226_1023"
    main(foldername)
