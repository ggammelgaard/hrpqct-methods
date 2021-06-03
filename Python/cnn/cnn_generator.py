import os
import copy
from random import shuffle
import itertools
import numpy as np
import nibabel as nib
from keras import backend as K
from keras.utils import to_categorical
from core.data import open_data_file
from core.generator import convert_data, get_number_of_steps
from core.augment import augment_data, random_permutation_x_y

from cnn.cnn_config import get_cnn_configuration
from core.model.tuber import tuber_model

K.set_image_dim_ordering('th')  # channels_first


def cnn_training_and_validation_generators(data_train_file, data_validation_file, batch_size,
                                           augment=False, augment_flip=True, augment_distortion_factor=0.25,
                                           validation_batch_size=None, skip_blank=True, permute=False):
    # EROSION = 1
    # CYST = 0
    training_list = list(range(data_train_file.root.data.shape[0]))  # needed for the data generator function
    validation_list = list(range(data_validation_file.root.data.shape[0]))  # needed for the datagenerator function

    training_generator = cnn_data_generator(data_train_file, training_list,
                                            batch_size=batch_size,
                                            augment=augment,
                                            augment_flip=augment_flip,
                                            augment_distortion_factor=augment_distortion_factor,
                                            skip_blank=skip_blank,
                                            permute=permute)

    validation_generator = cnn_data_generator(data_validation_file, validation_list,
                                              batch_size=validation_batch_size,
                                              skip_blank=skip_blank)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(len(training_list), batch_size)
    print("Number of training steps: ", num_training_steps)
    num_validation_steps = get_number_of_steps(len(validation_list), validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def add_r_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
               skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data = data_file.root.data[index]
    truth = data_file.root.data[index][0]  # not actually used for anything
    orig_label = data_file.root.label[index][0].decode('utf-8')
    if orig_label == 'erosion':
        label = 1
    else:
        label = 0

    if augment:
        affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])  # random permutation only accepts shape (n,X,Y,Z)
    else:
        truth = truth[np.newaxis]  # from here onward, truth has the same shape as data: (n,X,Y,Z)

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(label)


def convert_rdata(x_list, y_list):
    x = np.asarray(x_list)
    y = to_categorical(y_list, num_classes=2)  # onehot encoding

    return x, y


def cnn_data_generator(data_file, index_list, batch_size=1, augment=False, augment_flip=True,
                       augment_distortion_factor=0.25, shuffle_index_list=True, skip_blank=True,
                       permute=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()

        index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            # Add data to list
            add_r_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                       augment_distortion_factor=augment_distortion_factor, skip_blank=skip_blank, permute=permute)
            # Send data to algorithm
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_rdata(x_list, y_list)
                x_list = list()
                y_list = list()


def generator(train_file_path, validation_file_path, output_path):
    """ Only used for debugging purposes
    :param train_file_path:
    :param validation_file_path:
    :param output_path:
    :return:
    """
    # EROSION = 1
    # CYST = 0
    train_file = open_data_file(train_file_path)
    validation_file = open_data_file(validation_file_path)

    # Save image
    # i = 4
    # subject = data_file.root.subject_ids[i][0].decode('utf-8')
    # base_img = nib.Nifti1Image(data_file.root.data[i][0], data_file.root.affine[i])
    # base_img.to_filename(os.path.join(output_path, "tmp_{}_{}.nii.gz".format(i, subject)))
    # base_truth = nib.Nifti1Image(data_file.root.truth[i][0], data_file.root.affine[i])
    # base_truth.to_filename(os.path.join(output_path, "tmp_{}_{}_truth.nii.gz".format(i, subject)))

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = cnn_training_and_validation_generators(
        train_file,
        validation_file,
        batch_size=1,
        validation_batch_size=1,
        permute=False,  # should be enabled when i use cubic metrics
        augment=False,
        skip_blank=True,
        augment_flip=False)
    # test generator
    output1 = (next(train_generator))

    # Create model
    model = tuber_model()
    model.summary()

    # Start training
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=10,
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  max_q_size=2,
                                  )

    # Close data files
    train_file.close()
    validation_file.close()


if __name__ == '__main__':
    train_file_path = r'C:\school\repositories\au-rd-gustav-2020\code\machine_learning\RCNN\output_kfold\210226_1023\config_0_test_cc.h5'
    validation_file_path = r'C:\school\repositories\au-rd-gustav-2020\code\machine_learning\RCNN\output_kfold\210226_1023\config_0_validation_cc.h5'
    output_path = r'C:\school\RnD\trash'

    generator(train_file_path, validation_file_path, output_path)
