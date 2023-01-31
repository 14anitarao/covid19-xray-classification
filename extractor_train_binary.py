"""Train extractor and extract features."""

import os
import sys
import numpy as np
import keras
from keras import preprocessing
from utils import load_data, encode
from model import AdemNet, AdemNetV2, AdemNetV3, get_extractor, AdemNetV4Binary
from label_conf import SOURCE_PATH, NOFIN, COVID, PNEU, ARCH, AUGMENTED_PATH
from sklearn import metrics, model_selection
from datetime import datetime
import pandas as pd
import json


SIZE = 128
EPOCHS = 1000
BATCH_SIZE = 32
VAL_SPLIT = 0.2
CONTINUE = False


if __name__ == '__main__':
    try:
        size = int(sys.argv[1])
    except IndexError:
        print("No size argument given. Using default size", SIZE)
        size = SIZE
    except ValueError:
        print("Invalid size argument", sys.argv[1])
        sys.exit(1)
    print("Loading image data...")
    X, y, names = load_data(SOURCE_PATH, NOFIN[1], COVID[1], PNEU[1], SIZE)

    X = X[y!=2]
    names = [name for name, label in zip(names, y) if label != 2]
    y = y[y!=2]


    names = [name + f"_{label}" for name, label in zip(names, y)]
    print("Done.")
    print("Data shape:", X.shape[0])
    # X = X / 255.
    # X = X[y!=2]
    # y = y[y!=2]
    X_train, X_test, y_train, y_test, names_train, names_test = \
        model_selection.train_test_split(
            X, y, names, test_size=VAL_SPLIT, stratify=y)

    print("Training Data shape:", X_train.shape[0])

    if CONTINUE:
        model = keras.models.load_model(ARCH)
    else:
        model = AdemNetV4Binary()

    image_gen = preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # brightness_range=(-0.01, 0.01),
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        # validation_split=0.1,

        )
    model.fit(
        image_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test,y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=100,
                restore_best_weights=True
                )
            ]
        )
    model.summary()
    y_proba = model.predict(X_test)
    t_proba = model.predict(X_train)
    if y_proba.shape[1] > 1:
        y_pred = y_proba.argmax(axis=1)
        t_pred = t_proba.argmax(axis=1)
    else:
        y_pred = (y_proba > 0.5).astype(int).ravel()
        t_pred = (t_proba > 0.5).astype(int).ravel()

    cm_test = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred))
    cm_train = pd.DataFrame(metrics.confusion_matrix(y_train, t_pred))
    test_scores = {}
    test_scores['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    test_scores['f1_nofin'] = \
        metrics.f1_score(y_test, y_pred, pos_label=0)
    test_scores['f1_covid'] = \
        metrics.f1_score(y_test, y_pred, pos_label=1)
    test_scores['precision_nofin'] = \
        metrics.precision_score(y_test, y_pred, pos_label=0)
    test_scores['precision_covid']  = \
        metrics.precision_score(y_test, y_pred, pos_label=1)
    test_scores['recall_nofin'] = \
        metrics.recall_score(y_test, y_pred, pos_label=0)
    test_scores['recall_covid'] = \
        metrics.recall_score(y_test, y_pred, pos_label=1)
    test_scores['f1'] = metrics.f1_score(y_test, y_pred)
    test_scores['precision'] = \
        metrics.precision_score(y_test, y_pred)
    test_scores['recall'] = metrics.recall_score(y_test, y_pred)
    test_scores['auc'] = metrics.roc_auc_score(y_test, y_proba)

    train_scores = {}
    train_scores['accuracy'] = metrics.accuracy_score(y_train, t_pred)
    train_scores['f1_nofin'] = \
        metrics.f1_score(y_train, t_pred, pos_label=0)
    train_scores['f1_covid'] = \
        metrics.f1_score(y_train, t_pred, pos_label=1)
    train_scores['precision_nofin'] = \
        metrics.precision_score(y_train, t_pred, pos_label=0)
    train_scores['precision_covid'] = \
        metrics.precision_score(y_train, t_pred, pos_label=1)
    train_scores['recall_nofin'] = \
        metrics.recall_score(y_train, t_pred, pos_label=0)
    train_scores['recall_covid'] = \
        metrics.recall_score(y_train, t_pred, pos_label=1)
    train_scores['f1'] = metrics.f1_score(y_train, t_pred)
    train_scores['precision'] = \
        metrics.precision_score(y_train, t_pred)
    train_scores['recall'] = metrics.recall_score(y_train, t_pred)
    train_scores['auc'] = metrics.roc_auc_score(y_train, t_proba)

    print(cm_test)
    now = datetime.now().strftime('%Y%m%d%H%M')

    cm_test.to_csv(f'results/{ARCH}_binary_cm_test_{now}.csv')
    cm_train.to_csv(f'results/{ARCH}_binary_cm_train_{now}.csv')
    with open(f"results/{ARCH}_binary_test_scores_{now}.json", "w") as f:
        json.dump(test_scores, f)
    with open(f"results/{ARCH}_binary_train_scores_{now}.json", "w") as f:
        json.dump(train_scores, f)
    with open(f"filenames/filenames_train_binary_{now}.json", "w") as f:
        json.dump(names_train, f)
    with open(f"filenames/filenames_test_binary_{now}.json", "w") as f:
        json.dump(names_test, f)

    mapping = {
        v: k for k, v in {
            'No Finding': NOFIN[1],
            "Covid": COVID[1],
            "Pneumonia": PNEU[1]}.items()
        }

    extractor = get_extractor(model)

    e_train = extractor.predict(X_train).reshape(X_train.shape[0], -1)
    e_test = extractor.predict(X_test).reshape(X_test.shape[0], -1)
    e_train = pd.DataFrame(
        e_train, index=pd.Series(np.arange(e_train.shape[0]), name='image_id'))
    e_test = pd.DataFrame(
        e_test, index=pd.Series(np.arange(e_test.shape[0]), name='image_id'))
    e_train['label'] = y_train
    e_train['label_desc'] = e_train['label'].map(mapping)
    e_test['label'] = y_test
    e_test['label_desc'] = e_test['label'].map(mapping)

    e_train.to_csv(f"sets/train_binary_{now}.csv")
    e_test.to_csv(f"sets/test_binary_{now}.csv")

    model.save(ARCH + "Binary")
    print("Success.")
