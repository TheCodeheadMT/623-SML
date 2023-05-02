import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, RocCurveDisplay, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import SMLFinalPreprocessor as util

#### START: PREPROCESS DATA ################################################
preprocess = False
if preprocess:
    # read in dataset1.csv
    df = pd.read_csv("LABELED_2023_training_timeline1.txt", header=0,
                     names=['datetime', 'timestamp_desc', 'source', 'source_long',
                            'message', 'parser', 'display_name', 'tag'])

    # preprocess the timestamp field
    df.datetime = util.clean_datetimes(df.iloc[:, 0])

    # scale and encode the dataset
    df_transformed = util.transform_win7_traces(df)

    df_transformed.to_csv("LABELED_ENCODED_2023_training_timeline1.txt", index=False)
    exit()
############################################################################

# Decriptive data

# util.data_explore(df)
# print(df_transformed.head())
# df_transformed.drop("datetime")
# , figsize=(20,15)
histograms = False
if histograms:
    df_transformed.hist(layout=(1, 2), by='tag', range=(0, 1), figsize=(20, 15))
    plt.title("User Histograms")
    plt.show()

# show_histograms()


# sum columns and plot frequencies per class
# df_sums_user = df_transformed[df_transformed['tag'] == 1].sum()
# df_sums_sys = df_transformed[df_transformed['tag'] == 0].sum()
# df_sums_user.reindex(df_sums_sys.index)

# df_sums = df_sums.sort_values(by=[''])
# print(df_sums_user)
# print(df_sums_sys)
# df_sums_user.to_csv("df_user_sums.csv")
# df_sums_sys.to_csv("df_sys_sums.csv")
# df_sums_sys.hist(figsize=(20,15), alpha=.2, bins=10)
# df_sums_user.hist(figsize=(20,15), alpha=.6, bins=10)
# plt.title("User Feature Frequency")
# plt.show()

# df_sums_user


# sns.pairplot(df_transformed, plot_kws={'alpha': 0.5}) # I find the seaborn one to look a lot more polished than the pandas version out of the box
# plt.show();

# feature_freqs = df_transformed.sum(axis=1)
# print(feature_freqs)

# df_focused = df.drop(["datetime", ], axis=1)


####   END: PREPROCESS DATA ################################################
####### LOAD DATA ##########################################################
# read in dataset1.csv
df1 = pd.read_csv("LABELED_ENCODED_2023_training_timeline1.txt")

# Split data into two equal groups for training and validation maintaining class distributions
df1_train, df1_validation = train_test_split(df1, train_size=0.7, random_state=42, stratify=df1['tag'])

# Set weights for training
n_class_1 = df1_train[df1_train['tag'] == 1].shape[0]
n_class_0 = df1_train.shape[0] - n_class_1
n_samples = df1_train.shape[0]

#wj= n_samples / (n_classes * n_samplesj)

# Set class weights for training
class_weight = {
    0: n_samples/(2*n_class_0),
    1: n_samples/(2*n_class_1)
}
# print(class_weight)
# print(df1_train.shape)

# Set Xs and Ys for our training set
training = True
if training:
    X = df1_train.drop(["datetime", "tag"], axis=1)
    y = df1_train.tag
else:
    X = df1_validation.drop(["datetime", "tag"], axis=1)
    y = df1_validation.tag

# Get feature names
feature_names = X.columns.values.tolist()

# Build or load models
load_models = False
if load_models:
    if training:
        df1_train_models = util.load_all_models("training")
    else:
        df1_train_models = util.load_all_models("test")
else:
    df1_train_models = util.build_classifiers(X, y, class_weight, save_models=True, dataset_used="training")

# Print accuracy results
print_accuracy = True
if print_accuracy:
    print(util.report_model_accuracy(df1_train_models, X, y).to_string())

# Show confusion matrix for all models
show_confusion_matrix = True
if show_confusion_matrix:
    util.show_confusion_matrix(df1_train_models, X, y)

# Show ROC curves
show_roc_curves = True
if show_roc_curves:
    util.show_ROC_plots(df1_train_models, X, y)

# but our ROC curves are bad so  lets tune our hyperparameters
#util.gridSearchCV(df1_train_models, df1_validation)

# test against our test data



####### LOAD DATA ##########################################################

create_decision_tree_graph = False
if create_decision_tree_graph:
    dot_data = StringIO()
    export_graphviz(df1_train_models["Decision_Tree"], out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names = feature_names,
                    class_names=['0','1'])

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('userdata.png')
    Image(graph.create_png())

