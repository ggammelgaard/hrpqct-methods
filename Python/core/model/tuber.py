from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPool3D, \
    BatchNormalization, GlobalAveragePooling3D, Dense, Dropout, Flatten
from keras.engine import Model
from keras.optimizers import Adam
from keras.activations import softmax
from keras.backend import categorical_crossentropy
from keras import backend as K
from core.model.vggnet import get_optimizer_func

K.set_image_dim_ordering('th')  # channels_first


def tuber_model(input_shape=(1, 128, 128, 64), initial_learning_rate=5e-4, model_name="cnn_model.hd5",
                optimizer='adam', activation='softmax'):
    """
    https://keras.io/examples/vision/3D_image_classification/
    https://arxiv.org/pdf/2007.13224.pdf
    """
    inputs = Input(input_shape)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    # x = GlobalAveragePooling3D()(x)  # They use flatten in paper
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.6)(x)  # They use 0.6 in paper
    # x = Dropout(0.3)(x)

    outputs = Dense(units=2, activation=activation)(x)

    optimizer_func = get_optimizer_func(optimizer, initial_learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary(line_length=150)
    model.compile(optimizer=optimizer_func, loss="categorical_crossentropy", metrics=["acc"])
    model.save(model_name)
    return model
