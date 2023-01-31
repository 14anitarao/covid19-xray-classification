"""Resize and enumerate training samples."""

import cv2
import os
import sys
from label_conf import NOFIN, COVID, PNEU, DATA_PATH
from label_conf import SOURCE_PATH as OUT
import multiprocessing


if __name__ == '__main__':
    fi = 0
    for label in [NOFIN, COVID, PNEU]:
        filenames = [(fi, fname) for fi, fname in enumerate(os.listdir(os.path.join(DATA_PATH, label[0])))]
        for fi, fname in filenames:
            img = cv2.imread(os.path.join(DATA_PATH, label[0], fname), 0)
            img = cv2.resize(img, (120, 120))
            cv2.imwrite(os.path.join(OUT, str(label[1]), f"{fi:04d}.png"), img)
        # with multiprocessing.Pool(processes=12) as p:
        #     p.map(pro, filenames)
