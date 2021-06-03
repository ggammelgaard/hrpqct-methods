import os
import tables
import datetime
import glob
import json
import numpy as np
from random import shuffle
from shutil import copyfile

from core.data import write_data_to_file, open_data_file, create_data_file
from core.generator import get_unique_patient_finger_list
from unet.unet_config import kfold_root_path, get_kfold_configuration


def fetch_training_data_files(data_input_folder, return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in sorted(glob.glob(os.path.join(data_input_folder, "*")), key=str.lower):
        subject_ids.append(os.path.basename(subject_dir))  # i think this should be placed in the 'if len==2' statement
        subject_files = list()
        in_dir = os.listdir(os.path.join('.', subject_dir))
        if (len(in_dir) == 2):  # only adds data if there is two files in the folder (volume and label)
            in_dir.sort(key=str.lower)  # Sort files
            subject_files.append(os.path.join('.', subject_dir, in_dir[0]))
            subject_files.append(os.path.join('.', subject_dir, in_dir[1]))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def list_lengths(input_list):
    len_list = []
    for i in range(len(input_list)):
        try:
            len_list.append(len(np.hstack(input_list[i])))
        except ValueError:
            len_list.append(0)
    return len_list


def fill_label_lists(k, unique_list, unique_category, label_id):
    label_lists = [[] for _ in range(k)]
    erosion_category_indexes = list(np.where(np.array(unique_category) == label_id))[0].tolist()
    for index in erosion_category_indexes:
        fold_counts = list_lengths(label_lists)
        smallest_list_index = fold_counts.index(min(fold_counts))  # smallest group should have added a group of images
        label_lists[smallest_list_index].append(unique_list[index])
    return label_lists


def list_rotate(l, n):
    return l[n:] + l[:n]


def append_data(sender, receiver):
    # check if sender is empty
    if sender.root.data.shape[0] == 0:
        return
    # check if receiver is empty
    if receiver.root.data.shape[0] == 0:
        # copy content of sender to receiver
        sender.copy_node('/', newparent=receiver.root, overwrite=True)

    tx_data = sender.root.data
    tx_truth = sender.root.truth
    tx_affine = sender.root.affine
    tx_subject_ids = sender.root.subject_ids

    receiver.root.data.append(np.array(tx_data))
    receiver.root.truth.append(np.array(tx_truth))
    receiver.root.affine.append(np.array(tx_affine))
    receiver.root.subject_ids.append(np.array(tx_subject_ids))


def create_kfold_combinations(kconfig, foldername):
    # Find the k h5 files
    glob_list = glob.glob(os.path.join(kfold_root_path, foldername, 'kfold_split_*'))
    print("len(glob_list):", len(glob_list), "| kconfig['n_folds']:", kconfig["n_folds"])
    assert len(glob_list) == kconfig["n_folds"]  # we should only find files equal to the number of splits

    # make list of possible combinations (the order is what matters)
    subset_combinations = []
    subset_combinations.append(list(range(5)))
    for i in range(kconfig["n_folds"] - 1):
        subset_combinations.append(list_rotate(subset_combinations[i], 1))

    for i in range(kconfig["n_folds"]):
        print("Creating subfiles for combination {} of {}".format(i, kconfig["n_folds"] - 1))
        # create train subfile
        train_subfile1 = tables.open_file(glob_list[subset_combinations[i][0]], 'r')
        train_subfile2 = tables.open_file(glob_list[subset_combinations[i][1]], 'r')
        train_subfile3 = tables.open_file(glob_list[subset_combinations[i][2]], 'r')
        train_file = tables.open_file(os.path.join(kfold_root_path, foldername, 'config_{}_train.h5'.format(i)), 'w')
        # copy content of train_subfile1 to train_file
        train_subfile1.copy_node('/', newparent=train_file.root)
        # append remaining files
        append_data(train_subfile2, train_file)
        append_data(train_subfile3, train_file)

        # create validation subfile
        copyfile(glob_list[subset_combinations[i][3]],
                 os.path.join(kfold_root_path, foldername, 'config_{}_validation.h5'.format(i)))

        # create test subfile
        copyfile(glob_list[subset_combinations[i][4]],
                 os.path.join(kfold_root_path, foldername, 'config_{}_test.h5'.format(i)))

        # close opened files
        train_subfile1.close()
        train_subfile2.close()
        train_subfile3.close()
        train_file.close()


def create_kfolds(kconfig, foldername=None):
    # Local variables
    erosion_label = 3
    cyst_label = 2
    bone_label = 1
    bckgrnd_label = 0

    # Create output folder
    if not foldername:
        foldername = datetime.datetime.now().strftime("%y%m%d_%H%M")
    output_path = os.path.join(kfold_root_path, foldername)
    try:
        os.makedirs(output_path)
    except:
        pass

    # load all images
    training_files, subject_ids = fetch_training_data_files(kconfig["data_input"], return_subject_ids=True)
    # Create temporary datafile with all data loaded correctly
    total_data_path = os.path.join(output_path, "tmp_total_data.h5")
    write_data_to_file(training_files, total_data_path, image_shape=kconfig["image_shape"],
                       subject_ids=subject_ids,
                       crop=True)
    total_data_file = open_data_file(total_data_path)

    # give each image a label corresponding to if there is a cyst, erosion or none. Erosion has highest priority
    img_category = []  # Holds the category label for each image
    for i in range(total_data_file.root.data.shape[0]):  # for every image
        tmp_truth = total_data_file.root.truth[i]
        # count occurrences for each label
        values, counts = np.unique(tmp_truth, return_counts=True)
        content_dict = dict(zip(values, counts))
        # content_dict = dict()
        # for j in range(tmp_truth.shape[0]):
        #     count = np.count_nonzero(tmp_truth[j])
        #     if count != 0:
        #         content_dict[j + 1] = count

        if erosion_label in content_dict:  # Erosion has highest priority
            img_category.append(erosion_label)
        elif cyst_label in content_dict:
            img_category.append(cyst_label)
        else:
            img_category.append(bone_label)

    unique_list = get_unique_patient_finger_list(total_data_file)  # group images of the same finger into a list
    shuffle(unique_list)  # shuffle the list to add randomness
    unique_category = []
    # make category label for unique list
    for i in range(len(unique_list)):
        tmp_highest_category = 0
        for j in range(len(unique_list[i])):  # go through each image in each group
            category_of_current_image = img_category[unique_list[i][j]]
            if category_of_current_image > tmp_highest_category:
                tmp_highest_category = category_of_current_image
        unique_category.append(tmp_highest_category)  # give current group the highest category of its contents

    assert len(unique_list) == len(unique_category)

    # place equal amount of erosion, cyst, and bone images in k lists
    erosion_lists = fill_label_lists(kconfig["n_folds"], unique_list, unique_category, erosion_label)
    cyst_lists = fill_label_lists(kconfig["n_folds"], unique_list, unique_category, cyst_label)
    bone_lists = fill_label_lists(kconfig["n_folds"], unique_list, unique_category, bone_label)

    flat_erosion_list = []
    flat_cyst_list = []
    flat_bone_list = []
    # save sets in 5 hdf5 files
    for i in range(kconfig["n_folds"]):
        # flatten index lists, as the finger groups are not used from here on (they were just for distribution)
        try:
            flat_erosion_list.append(np.array(np.hstack(erosion_lists[i])).tolist())
        except ValueError:
            flat_erosion_list.append([])
        try:
            flat_cyst_list.append(np.array(np.hstack(cyst_lists[i])).tolist())
        except ValueError:
            flat_cyst_list.append([])
        try:
            flat_bone_list.append(np.array(np.hstack(bone_lists[i])).tolist())
        except ValueError:
            flat_bone_list.append([])

        # create a combined list of all the sublists
        combined_list = flat_erosion_list[i] + flat_cyst_list[i] + flat_bone_list[i]
        shuffle(combined_list)

        # create .h5 file
        filename = "kfold_split_{}.h5".format(i)
        sub_file_path = os.path.join(output_path, filename)
        n_samples = len(erosion_lists[i]) + len(cyst_lists[i]) + len(bone_lists[i])
        n_truth_labels = 1

        sub_file, data_storage, truth_storage, affine_storage = create_data_file(sub_file_path,
                                                                                 n_channels=1,
                                                                                 n_samples=n_samples,
                                                                                 n_truth_labels=n_truth_labels,
                                                                                 image_shape=kconfig["image_shape"])

        # fill .h5 file
        for index in combined_list:
            data_storage.append(total_data_file.root.data[index][np.newaxis])
            truth_storage.append(total_data_file.root.truth[index][np.newaxis])
            affine_storage.append(total_data_file.root.affine[index][np.newaxis])
        # adding subject_ids afterwards because they did so in original code
        subject_sublist = [subject_ids[index] for index in combined_list]
        sub_file.create_earray(sub_file.root, 'subject_ids', obj=subject_sublist)
        # Close subfile
        sub_file.close()

    # Close data file
    total_data_file.close()
    os.remove(total_data_path)  # not needed anymore, and quite large

    # Create logfile
    kfold_log = {}
    kfold_log['k'] = kconfig["n_folds"]
    kfold_log['data_input_folder'] = kconfig["data_input"]
    kfold_log['erosion_volume_count'] = [len(x) for x in flat_erosion_list]
    kfold_log['cyst_volume_count'] = [len(x) for x in flat_cyst_list]
    kfold_log['bone_volume_count'] = [len(x) for x in flat_bone_list]
    kfold_log['summed_volume_count'] = [sum(x) for x in
                                        zip(kfold_log['erosion_volume_count'], kfold_log['cyst_volume_count'],
                                            kfold_log['bone_volume_count'])]
    kfold_log['erosion_volume_ratio'] = np.divide(kfold_log['erosion_volume_count'],
                                                  kfold_log['summed_volume_count']).tolist()
    kfold_log['cyst_volume_ratio'] = np.divide(kfold_log['cyst_volume_count'],
                                               kfold_log['summed_volume_count']).tolist()
    kfold_log['bone_volume_ratio'] = np.divide(kfold_log['bone_volume_count'],
                                               kfold_log['summed_volume_count']).tolist()
    kfold_log['unique_fingers'] = len(unique_list)
    kfold_log['total_volumes'] = sum(kfold_log['summed_volume_count'])
    kfold_log["kfold_path"] = output_path

    with open(os.path.join(output_path, 'kfold_log.json'), 'w') as outfile:
        json.dump(kfold_log, outfile, indent=4, sort_keys=True)

    return output_path


def main(foldername):
    # Get configuration
    kconfig = get_kfold_configuration(foldername)

    # Create data
    output_path = create_kfolds(kconfig, foldername)
    create_kfold_combinations(kconfig, foldername)

    # Save configuration
    with open(os.path.join(output_path, 'kconfig.json'), 'w') as outfile:
        json.dump(kconfig, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    foldername = "experiment05"
    main(foldername)
