import sys
import os
import numpy as np


def weight_calculator(hnh=False):
    # project imports
    sys.path.insert(0, os.path.dirname(os.path.abspath('')))
    from core.data import write_data_to_file, open_data_file
    from unet.unet_config import data_root_path
    from unet.kfold import fetch_training_data_files

    # Decide on variables
    data_path = os.path.join(data_root_path, "skejby2020_0")  # skejby2020_0

    image_shape = (128, 128, 128)
    if not hnh:
        n_labels = 4
    else:
        n_labels = 2

    # Label variables
    nonhealthy_idx = 1
    erosion_label = 3
    cyst_label = 2
    bone_label = 1
    bckgrnd_label = 0

    # load all images
    training_files, subject_ids = fetch_training_data_files(data_path, return_subject_ids=True)
    # Create temporary datafile with all data loaded correctly
    total_data_path = "tmp_total_data.h5"
    write_data_to_file(training_files, total_data_path, image_shape=image_shape,
                       subject_ids=subject_ids,
                       crop=True)
    total_data_file = open_data_file(total_data_path)

    # Loop through each image
    overlap_count = 0
    N = total_data_file.root.data.shape[0]
    label_area = [[] for i in range(n_labels)]
    for i in range(N):  # for every image
        tmp_truth = total_data_file.root.truth[i]
        # count occurrences for each label
        values, counts = np.unique(tmp_truth, return_counts=True)
        content_dict = dict(zip(values, counts))
        # fill counts into each corresponding sublist
        if not hnh:
            if erosion_label in content_dict:
                label_area[erosion_label].append(content_dict[erosion_label])
                if cyst_label in content_dict:  # If both cyst and erosion are present
                    overlap_count += 1
            if cyst_label in content_dict:
                label_area[cyst_label].append(content_dict[cyst_label])
            if bone_label in content_dict:
                label_area[bone_label].append(content_dict[bone_label])
            if bckgrnd_label in content_dict:
                label_area[bckgrnd_label].append(content_dict[bckgrnd_label])

                if not bone_label in content_dict:  # find rogue volume with only background
                    print("EMPTY VOLUME!!!: {}".format(total_data_file.root.subject_ids[i]))

        else:
            tmp_nonhealthy_count = 0
            tmp_healthy_count = 0
            if erosion_label in content_dict:
                tmp_nonhealthy_count += content_dict[erosion_label]
            if cyst_label in content_dict:
                tmp_nonhealthy_count += content_dict[cyst_label]
            if bone_label in content_dict:
                tmp_healthy_count += content_dict[bone_label]
            if bckgrnd_label in content_dict:
                tmp_healthy_count += content_dict[bckgrnd_label]

            if tmp_nonhealthy_count != 0:
                label_area[nonhealthy_idx].append(tmp_nonhealthy_count)
            if tmp_healthy_count != 0:
                label_area[bckgrnd_label].append(tmp_healthy_count)

    # Find area weights
    occurences_count = [len(x) for x in label_area]
    frequency_weight = [N / len(x) for x in label_area]
    area_weight = [np.sum(np.sum(label_area)) / sum(x) for x in label_area]

    # Print
    print("frequency_weight: {}".format(["{:.2f}".format(i) for i in frequency_weight]))
    print("area_weight:      {}".format(["{:.2f}".format(i) for i in area_weight]))
    print()
    print("occurences_count: {}".format(["{:.2f}".format(i) for i in occurences_count]))
    print("cyst+erosion overlap: {}".format(overlap_count))

    # Close data file
    total_data_file.close()
    os.remove(total_data_path)  # not needed anymore, and quite large


if __name__ == '__main__':
    weight_calculator(hnh=False)
