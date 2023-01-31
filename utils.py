import os
import cv2
import numpy as np
from typing import List

def load_image(path, size):
    return cv2.resize(cv2.imread(path, 0), size) / 255.


def load_data(source_path: str, nofind_label: int, covid_label: int, pneu_label: int, size=128) -> (np.ndarray, np.ndarray, List[str]):
    """Load image data.

    Args:
    -----------
    source_path : str, main image directory
    nofind_label: int, No_findings directory
    covid_label: int, COVID directory
    pneu_label: int, Pneumonia directory
    size: int, image reshape size

    Returns:
    -----------
    X : np.ndarray, (M x size x size) training images
    y : np.ndarray, (M,) training labels
    names : List[str], filenames
    """

    names = []
    nofinddata = []
    for file in os.listdir(os.path.join(source_path, str(nofind_label))):
        nofinddata.append(
            load_image(
                os.path.join(source_path, str(nofind_label), file),
                (size, size)
                ).reshape(size, size, 1)
            )
        names.append(file)

    coviddata = []
    for file in os.listdir(os.path.join(source_path, str(covid_label))):
        coviddata.append(
            load_image(
                os.path.join(source_path, str(covid_label), file),
                (size, size)
                ).reshape(size, size, 1)
            )
        names.append(file)
    pneudata = []
    for file in os.listdir(os.path.join(source_path, str(pneu_label))):
        pneudata.append(
            load_image(
                os.path.join(source_path, str(pneu_label), file),
                (size, size)
                ).reshape(size, size, 1)
            )
        names.append(file)

    X = np.array(nofinddata + coviddata + pneudata)
    y = np.array([nofind_label] * len(nofinddata) + [covid_label] * len(coviddata) + [pneu_label] * len(pneudata))

    return X, y, names


def process(images):
    return images / 255.


def encode(y, num_classes=2):
    return (y.reshape(-1, 1) == np.arange(num_classes + 1).reshape(1, -1))
