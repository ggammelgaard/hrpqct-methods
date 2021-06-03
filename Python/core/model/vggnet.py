from keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Dense, Dropout, GlobalAveragePooling3D, Flatten
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.activations import softmax
from keras.backend import categorical_crossentropy
from keras import backend as K

K.set_image_dim_ordering('th')  # channels_first


def get_optimizer_func(optimizer_string, learning_rate):
    if optimizer_string == 'adam':
        return Adam(lr=learning_rate)
    elif optimizer_string == 'sgd_nesterov':
        return SGD(lr=learning_rate, nesterov=True)
    elif optimizer_string == 'sgd_momentum':
        return SGD(lr=learning_rate, momentum=0.99)
    else:
        print("ERROR! NO OPTIMIZER FUNC FOUND")
        return None


def vggnet_model(input_shape=(1, 110, 110, 110), initial_learning_rate=5e-4, model_name="cnn_model.hd5",
                 optimizer='adam', activation='softmax'):
    """
    http://arxiv.org/abs/1701.06643
    """
    assert input_shape == (1, 110, 110, 110)
    inputs = Input(input_shape)

    x = Conv3D(filters=8, kernel_size=3, activation="relu")(inputs)
    x = Conv3D(filters=8, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(units=128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = Dense(units=64, activation="relu")(x)

    outputs = Dense(units=2, activation=activation)(x)

    optimizer_func = get_optimizer_func(optimizer, initial_learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_func, loss="categorical_crossentropy", metrics=["acc"])
    model.save(model_name)
    return model
