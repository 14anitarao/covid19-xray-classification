import os

NOFIN = ("No_findings", 0)
COVID = ("covid-19", 1)
PNEU = ("Pneumonia", 2)

DATA_PATH = "expanded-dataset"
SOURCE_PATH = os.path.join("img", "processed")
AUGMENTED_PATH = os.path.join("img", "augmented")
HIGHLIGHT_PATH = os.path.join("img", "highlighted")
EXTRACTED_OUT = "extracted"
ARCH = "AdemNetV4"
