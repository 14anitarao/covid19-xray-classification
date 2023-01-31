# Team8_Covid19
Contributors:
- Anita Rao
- Ilker Yaramis
- Sukru Sezer

## Environment Setup
1. Clone Github repository
2. Install Python3 (PyCharm IDE Recommended)
  1. `python --version`
  2. [Python Official](https://www.python.org/downloads/) (link)
  3. Setup environment
    - `pip install -r requirements.txt`
3. Install Scala (IntelliJ IDE Recommended)
  1. [IntelliJ Idea for Scala](https://www.jetbrains.com/help/idea/discover-intellij-idea-for-scala.htm) (link)


```
Team8_Covid19
├── AdemNetV4
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── AdemNetV4Binary
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── README.md
├── app_requirements.txt
├── covid_19_detection_app.py
├── doPCA.py
├── expanded-dataset
│   ├── No_findings
│   ├── Pneumonia
│   └── covid-19
├── extractor_train.py
├── extractor_train_binary.py
├── filenames
├── img
│   └── processed
│       ├── 0
│       ├── 1
│       └── 2
├── label_conf.py
├── mlTraining.scala
├── model.py
├── preprocess.py
├── requirements.txt
├── results
├── sets
├── train5.ps1
└── utils.py
```

## Data Sources
The following two sources were used in this project
<ol>
<li>COVID-Chest-xray-dataset
<ol>
<li>https://github.com/ieee8023/covid-chestxray-dataset</li>
</ol>  
<li>COVID-19 dataset</li>
<ol>
<li>https://github.com/muhammedtalo/COVID-19</li>
</ol>
</ol>


- Normal and pneumonia images:
  - [ChestX-ray8 Database](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
- COVID-19+ Images:
  - [Cohen, J.P. Database](https://github.com/ieee8023/covid-chestxray-dataset)

## Pre-processing & Feature Extraction Steps

## Model Training

### Machine Learning Models
<ol>
<li>Run <code>python3 doPCA.py</code></li>
<li>5 train/test CSV pairs (10 files) should be generated in the current working directory: </li>
<ol>
<li>train0.csv and test0.csv</li>
<li>train1.csv and test1.csv</li>
<li>train2.csv and test2.csv</li>
<li>train3.csv and test3.csv</li>
<li>train4.csv and test4.csv</li>
</ol>
<li>Run <code>mlTraining.scala</code></li>
<li>Performance metrics (accuracy, precision, recall, f1score) for Logistic Regression and Random Forest models should be printed in the console</li>
</ol>


- Training and evaluating the model:
  - For a single training and evaluation of a binary classifier:
    - `python extractor_train_binary.py`
  - For a single training and evaluation of a multi-class classifier:
    - `python extractor_train.py`
  - For Monte Carlo Cross Validation:
    - `train5.ps1` (Windows) or `train5.sh` (Linux/UNIX)
  - Performance evaluation results will be dumped to file `results/AdemNetV4_<train|test>_scores_<YYYYmmddHHMM>.json`
  - 5 pairs of training/test datasets of extracted features will be generated in `sets/<train|test>_<yyyyMMddHHMM>.`

### COVID-19 Detection Application
- Run application server:
  - `python covid-covid_19_detection_app.py`

