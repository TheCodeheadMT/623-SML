import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as col

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import SMLFinalPreprocessor as util

#### START: PREPROCESS DATA ################################################
# read in dataset1.csv
df = pd.read_csv("timeline1.txt", header=0,
                  names=['datetime','timestamp_desc','source','source_long',
                         'message','parser','display_name','tagX1'], index_col=0)

clean_time_ndarray = util.clean_datetimes(df.iloc[:, 0])



####   END: PREPROCESS DATA ################################################
####### LOAD DATA ##########################################################
# read in dataset1.csv
df1 = pd.read_csv("timeline1.txt", header=0,
                  names=['datetime','timestamp_desc','source','source_long',
                         'message','parser','display_name','tagX1'], index_col=0)

# Split data into two equal groups for training and testing maintaining class distributions
df1_train, df1_test = train_test_split(df1, test_size=0.5, random_state=42)

print(df1_train)

####### LOAD DATA ##########################################################
