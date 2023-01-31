import keras
from keras import models, layers
import cv2
import numpy as np

def AdemNet():
    """AdemNet."""
    model = models.Sequential()
    model.add(layers.Conv2D(4, (1, 3), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(layers.Conv2D(4, (3, 1), activation='relu', name='conv0'))
    model.add(layers.MaxPool2D((2,2)))
#     model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(4, (1, 3), activation='relu'))
    model.add(layers.Conv2D(4, (3, 1), activation='relu', name='conv1'))
    model.add(layers.MaxPool2D((2,2)))
#     model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(4, (1, 3), activation='relu'))
    model.add(layers.Conv2D(4, (3, 1), activation='relu', name='conv2'))
    model.add(layers.MaxPool2D((2,2)))
#     model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(1, (1, 3), activation='relu'))
    model.add(layers.Conv2D(1, (3, 1), activation='relu', name='extracted'))
#     model.add(keras.layers.GlobalMaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(8, activation='relu'))
#     model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='relu'))
#     model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            ]
        )
    return model


def AdemNetV2():
    """AdemNetV2."""
    model = models.Sequential()
    model.add(layers.Conv2D(12, (7, 7), activation=None, kernel_initializer='glorot_uniform', input_shape=(128, 128, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(24, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(36, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(48, (3, 3), activation=None, name='extracted'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(84, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            ]
        )
    return model


def AdemNetV4():
    model = models.Sequential()
    model.add(layers.Conv2D(24, (7, 7), activation=None, kernel_initializer='glorot_uniform', input_shape=(128, 128, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None, name='extracted'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(84, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            ]
        )
    return model


def AdemNetV4Binary():
    model = models.Sequential()
    model.add(layers.Conv2D(24, (7, 7), activation=None, kernel_initializer='glorot_uniform', input_shape=(128, 128, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(124, (3, 3), activation=None, name='extracted'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D((2,2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(84, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            ]
        )
    return model


def get_extractor(model: keras.models.Model):
    """Return an extractor from a trained AdemNet model.

    Extraction layer must be explicitly named as `extracted`.
    """
    layer_names = {model.layers[i].name: i for i in range(len(model.layers))}
    input_ = model.input
    output_ = model.layers[layer_names['extracted']]
    aux_model = keras.models.Model(inputs=input_, outputs=output_.get_output_at(0))
    return aux_model


def get_predictor(model: keras.models.Model):
    """Return a predictor from a trained AdemNet model.

    Extraction layer must be explicitly named as `extracted`.
    """
    layer_names = {model.layers[i].name: i for i in range(len(model.layers))}

    input_ = model.layers[layer_names['extracted']]
    output_ = model.output
    aux_model = keras.models.Model(inputs=input_.get_input_at(0), outputs=output_)
    return aux_model


def relu(x):
    x = x.copy()
    x[x<0.] = 0.
    return x


def visualize_decision(img, extractor, predictor, size=256, sharpen=1):
    A = extractor.predict(img.reshape(1,size,size,1))
    y_proba = predictor.predict_proba(A.reshape(1, -1))
    try:
        W = predictor.named_steps['clf'].coef_[y_proba.argmax()]
    except KeyError:
        raise ValueError("Predictor object must be a pipeline and classifier must be named as `clf`.")
    y_pred = y_proba[0][y_proba.argmax()]
    Ws = W.reshape(A.shape).squeeze()
    a = []
    for k in range(Ws.shape[2]):
        a.append(((y_pred * (1 - y_pred) * Ws[:,:,k])).mean())
    visualized = cv2.resize(relu((np.array(a) * Ws).sum(axis=2)), (size, size))**sharpen
    return visualized
