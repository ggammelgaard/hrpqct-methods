from keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Dense, Dropout, GlobalAveragePooling3D, Flatten, \
    ReLU, Add
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.activations import softmax
from keras.backend import categorical_crossentropy
from keras import backend as K
from core.model.vggnet import get_optimizer_func

K.set_image_dim_ordering('th')  # channels_first


def resnet_model(input_shape=(1, 110, 110, 110), initial_learning_rate=5e-4, model_name="cnn_model.hd5",
                 optimizer='adam', activation='softmax'):
    """
    https://arxiv.org/abs/1803.02544
    https://github.com/neuro-ml/resnet_cnn_mri_adni/blob/master/scripts/resnet_train.ipynb
    """
    assert input_shape == (1, 110, 110, 110)
    inputs = Input(input_shape)

    # block 1
    conv1a = Conv3D(filters=32, kernel_size=3, padding='same')(inputs)
    bn1a = BatchNormalization()(conv1a)
    relu1a = ReLU()(bn1a)
    conv1b = Conv3D(filters=32, kernel_size=3, padding='same')(relu1a)
    bn1b = BatchNormalization()(conv1b)
    relu1b = ReLU()(bn1b)
    conv1c = Conv3D(filters=64, kernel_size=3, padding='same', strides=2)(relu1b)
    # VoxRes block 2
    voxres2_bn1 = BatchNormalization()(conv1c)
    voxres2_relu1 = ReLU()(voxres2_bn1)
    voxres2_conv1 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres2_relu1)
    voxres2_bn2 = BatchNormalization()(voxres2_conv1)
    voxres2_relu2 = ReLU()(voxres2_bn2)
    voxres2_conv2 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres2_relu2)
    voxres2_out = Add()([conv1c, voxres2_conv2])
    # VoxRes block 3
    voxres3_bn1 = BatchNormalization()(voxres2_out)
    voxres3_relu1 = ReLU()(voxres3_bn1)
    voxres3_conv1 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres3_relu1)
    voxres3_bn2 = BatchNormalization()(voxres3_conv1)
    voxres3_relu2 = ReLU()(voxres3_bn2)
    voxres3_conv2 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres3_relu2)
    voxres3_out = Add()([voxres2_out, voxres3_conv2])
    # block 4
    bn4 = BatchNormalization()(voxres3_out)
    relu4 = ReLU()(bn4)
    conv4 = Conv3D(filters=64, kernel_size=3, padding='same', strides=2)(relu4)
    # VoxRes block 5
    voxres5_bn1 = BatchNormalization()(conv4)
    voxres5_relu1 = ReLU()(voxres5_bn1)
    voxres5_conv1 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres5_relu1)
    voxres5_bn2 = BatchNormalization()(voxres5_conv1)
    voxres5_relu2 = ReLU()(voxres5_bn2)
    voxres5_conv2 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres5_relu2)
    voxres5_out = Add()([conv4, voxres5_conv2])
    # VoxRes block 6
    voxres6_bn1 = BatchNormalization()(voxres5_out)
    voxres6_relu1 = ReLU()(voxres6_bn1)
    voxres6_conv1 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres6_relu1)
    voxres6_bn2 = BatchNormalization()(voxres6_conv1)
    voxres6_relu2 = ReLU()(voxres6_bn2)
    voxres6_conv2 = Conv3D(filters=64, kernel_size=3, padding='same')(voxres6_relu2)
    voxres6_out = Add()([voxres5_out, voxres6_conv2])
    # block 7
    bn7 = BatchNormalization()(voxres6_out)
    relu7 = ReLU()(bn7)
    conv7 = Conv3D(filters=128, kernel_size=3, padding='same', strides=2)(relu7)
    # VoxRes block 8
    voxres8_bn1 = BatchNormalization()(conv7)
    voxres8_relu1 = ReLU()(voxres8_bn1)
    voxres8_conv1 = Conv3D(filters=128, kernel_size=3, padding='same')(voxres8_relu1)
    voxres8_bn2 = BatchNormalization()(voxres8_conv1)
    voxres8_relu2 = ReLU()(voxres8_bn2)
    voxres8_conv2 = Conv3D(filters=128, kernel_size=3, padding='same')(voxres8_relu2)
    voxres8_out = Add()([conv7, voxres8_conv2])
    # VoxRes block 9
    voxres9_bn1 = BatchNormalization()(voxres8_out)
    voxres9_relu1 = ReLU()(voxres9_bn1)
    voxres9_conv1 = Conv3D(filters=128, kernel_size=3, padding='same')(voxres9_relu1)
    voxres9_bn2 = BatchNormalization()(voxres9_conv1)
    voxres9_relu2 = ReLU()(voxres9_bn2)
    voxres9_conv2 = Conv3D(filters=128, kernel_size=3, padding='same')(voxres9_relu2)
    voxres9_out = Add()([voxres8_out, voxres9_conv2])
    # ending
    pool10 = MaxPool3D(pool_size=7)(voxres9_out)
    flat10 = Flatten()(pool10)
    dense11 = Dense(units=128, activation="relu")(flat10)
    outputs = Dense(units=2, activation=activation)(dense11)

    optimizer_func = get_optimizer_func(optimizer, initial_learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_func, loss="categorical_crossentropy",
                  metrics=["acc"])
    model.save(model_name)
    return model
