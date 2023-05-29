import datetime as datetime
import itertools
import os
import pickle
import time
from io import StringIO
from pathlib import Path

import pydotplus
from IPython.display import Image

import numpy as np
import pandas as pd
#import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from keras import Model
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
# from keras.models import KerasClassifier
# from sklearn.keras.wrappers.scikit_learn.KerasClassifier,
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


# Given a date time str, convert it to unixtime
def fix_datetime(datetime_str):
    # 1601-01-01T00:00:00.307030+00:00
    if datetime_str.split("-")[0] == "0000":
        return 0.0
    utc_dt = datetime.datetime.strptime(datetime_str.split("+")[0], '%Y-%m-%dT%H:%M:%S.%f')
    timestamp = (utc_dt - datetime.datetime(1970, 1, 1)).total_seconds()
    if timestamp < 0:
        return 0.0
    return timestamp


# Given datetime column, apply a mapping to column to convert it to unix time
def clean_datetimes(datetime_col):
    return datetime_col.apply(fix_datetime)


def corr_plots(df, title):
    plt.yticks(rotation=45)
    # plotting the heatmap
    # Show heatmap of our correlation matrix
    corr_vals = df.corr().sort_values('tag', ascending=False)
    dataplot = sns.heatmap(corr_vals, cmap="YlGnBu", annot=True)
    plt.yticks(rotation=25)
    plt.xticks(rotation=25)
    plt.title(title)
    plt.show()
    plt.savefig("figures\\heatmap_corr.png")


def box_plots(df):
    # reverse one-hot encoding
    # df[] = (df.iloc[:, 2:] == 1).idxmax(1)

    # actuals and predicted to one column
    #df = df.melt(id_vars ='msg_len', value_vars=['file_depth', 'msg_len', 'source', 'source_long', 'key_dirs'])
    df1 = df[['file_depth', 'msg_len', 'source', 'source_long', 'key_dirs']]
    #df1['msg_len' > 600] = np.mean(df1['msg_len'])
    df1_melt = pd.melt(df1)

    sns.boxplot(x='variable', y='value', hue='variable', data=df1_melt, palette="Set3")

    #plt.ylim([0, 50])
    plt.yscale('log')

    plt.show()
    return df


def data_explore_msg_len_file_depth(df):

    for ftr in df:
        # print description of this feature
        print(ftr, " Description:\n", df[ftr].describe(), '\n')

    df1 = df[df['tag'] == 1]
    df0 = df[df['tag'] == 0]

    bincount = 20
    bins_file_msg_len = np.linspace(0, 2691, bincount)
    # show a historgram for each class 0, 1
    plt.figure(1)
    plt.hist(df0['msg_len'], bins_file_msg_len, label="System msg_len", alpha=.25)
    plt.hist(df1['msg_len'], bins_file_msg_len, label="User msg_len", alpha=.25)
    plt.legend(loc='upper right')
    plt.title("System vs. User Message Length")
    plt.yscale('log')
    #plt.xscale('log')
    plt.show()

    plt.figure(2)
    bins_file_depth = np.linspace(0, 20, bincount)
    # show a historgram for each class 0, 1
    #plt.axis([0, 18, 0, 10])
    plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df0['file_depth'], bins_file_depth, label="System file_depth", alpha=.25)
    plt.hist(df1['file_depth'], bins_file_depth, label="User file_depth", alpha=.25)
    plt.legend(loc='upper right')
    plt.title("System vs. User File Depth")
    plt.show()

    # # show covariance for class 1
    print("Covariance Tag 1 (user):\n", df[df.tag == 1].cov(), '\n')

    # show covariance for class 0
    print("Covariance Tag 0 (system):\n", df[df.tag == 0].cov(), '\n')


def transform_win7_traces(features_df):
    with Timer("Transform Win 7 Traces"):
        features_out = features_df
        tags = features_df.tag

        # features not used at all in model
        primary_exclude_list = ['message', 'display_name', 'parser', 'tag']

        # nominal features used in model
        # nom_features_list = ['timestamp_desc', 'source', 'source_long']
        nom_features_list = ['source', 'source_long', 'timestamp_desc', 'key_dirs']
        # 'first_dir'

        # numeric features used in model
        num_features_list = ['datetime', 'msg_len', 'file_depth']

        # Removed 'flagged_activity_dist', 'flagged_activity_dist_X_file_depth'

        # Remove columns we do not want to use...
        features_out = features_out.drop(primary_exclude_list, axis=1)

        # Initial fields included
        # Nominal: timestamp_desc, source, source_long
        # Scalar: datetime

        # FEATURES SCALED ################################################################
        # remove nominal features
        features_num = features_out.drop(nom_features_list, axis=1)

        # init standard scalar
        scaler = StandardScaler()

        # fit and transform our data
        scaled = scaler.fit_transform(features_num)

        # convert back to a DF
        scaled_data = pd.DataFrame(scaled)

        # recover headers
        scaled_data.columns = features_num.columns

        # update feature_num state
        features_num = scaled_data

        # FEATURES ENCODED ##############################################################
        # get only nominal features by dropping numeric cols
        features_nom = features_out.drop(num_features_list, axis=1)

        # get 1-hot encoding for nominal features
        features_nom = pd.get_dummies(features_nom)

        # combine scaled and nominal features into one df
        features_num.reset_index(drop=True, inplace=True)
        features_nom.reset_index(drop=True, inplace=True)
        tags.reset_index(drop=True, inplace=True)

        features_out = pd.concat([features_num, features_nom, tags], axis=1)
        df_out = pd.DataFrame(data=features_out,
                              index=features_out.index,
                              columns=features_out.columns)

        # This feature is not in the test data, so must not be included in training.
        # df_out = df_out.drop("source_long_Chrome Cache", axis=1)

    return df_out


def parse_first_dir_from_display_name_feature(df):
    with Timer("Add feature - root directory"):  # get first directory from display name
        df['first_dir'] = df.apply(
            lambda x: (x['display_name'].split('\\')[1])
            if ('\\' in x['display_name']) and
               not ('$' in x['display_name'].split('\\')[1]) else "\\", axis=1)

    return df


def get_msg_len_feature(df):
    with Timer("Add feature - message length"):
        df['message'] = df['message'].astype(str)
        df['msg_len'] = df.apply(lambda x: int(len(x['message'])), axis=1)

    return df


def get_msg_len_feature_src_long(df, src_long_list):
    with Timer("Add feature - src long message length"):
        for idx, row in df.itterrows():
            df['message'] = df['message'].astype(str)
            df['msg_len'] = df.apply(lambda x: int(len(x['message'])), axis=1)

    return df


def get_time_delta_from_tagged_activity_feature(df):
    # build a hashmap of tags and timestamps
    # if tag is 1, set value to 0
    # if tag is 0, set value to closest 1 timestamp - current obseravtion timestamp
    # fir each entry: find the next entry that
    with Timer("Add feature - delte from user activity"):
        df['flagged_activity_dist'] = 10e+100

        # forward pass
        for index, row in df.iterrows():
            # if row['tag'] == 1:
            #     df.at[index, 'flagged_activity_dist'] = 0
            # if row['tag'] != 1:
            # check forward
            t_past = None
            t_current = None  # the last time entry I saw
            t_future = None  # the next time entry I see
            for i in range(df.shape[0] - index + 1):
                # set my spot to the current observation timestamp
                i += index
                if i > df.shape[0] - 1:
                    break
                # print("starting index", i)
                if df.at[i, 'tag'] == 1:
                    df.at[index, 'flagged_activity_dist'] = df.at[i, 'datetime'] - row[0]
                    break
            # check backward
            for j in range(index, 0, -1):
                # set my spot to the current observation timestamp
                if j < 0:
                    break
                # print("starting index", i)
                if df.at[j, 'tag'] == 1:
                    if df.at[j, 'datetime'] - row[0] < df.at[index, 'flagged_activity_dist']:
                        df.at[index, 'flagged_activity_dist'] = np.abs(df.at[j, 'datetime'] - row[0])
                    break

            if index % 10000 == 0:
                print("-> % Complete: ", index / df.shape[0])

        # add one so there is no 0, for interaction use later.
        df['flagged_activity_dist'] = df['flagged_activity_dist'] + 1

    return df


# helper function for encode_display_name_feature
def encode_display_name(dispplay_str):
    out_str = ''
    p = Path(dispplay_str)
    key_dirs = ['windows', 'program files', 'users', 'windows', 'system32', 'common files', 'documents', 'pictures',
                'appdata', 'roaming', 'local settings', 'application data', 'local', 'desktop',
                'documents and settings']
    # excluded 'temp'

    p_str = os.path.splitext(p)[0].lower()

    for idx, dir in enumerate(key_dirs):
        if idx > 5:
            out_str = out_str + "\\*"
        else:
            if dir in p_str:
                out_str = out_str + "\\" + dir + "\\*"
            else:
                out_str = out_str + "\\*"
    return out_str


# transform display_name column into simplified represenatation
def encode_display_name_feature(df):
    with Timer("Add feature - encoded display name"):
        df['key_dirs'] = df['display_name'].apply(encode_display_name)
    return df


def get_file_depth_feature(display_name_str):
    p = Path(display_name_str)
    return len(p.parts)


def encode_file_depth_feature(df):
    with Timer("Add feature - encode file depth feature"):
        df['file_depth'] = df['display_name'].apply(get_file_depth_feature)
    return df


def build_classifiers(Xs, ys, class_weights, save_models=False, dataset_used="training"):
    X = Xs
    y = ys

    with Timer("Build and Fit: Logistic Regression"):
        log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, C=0.001, penalty='l2')
        #log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
        log_reg.fit(X, y)

    # with Timer("Build and Fit: SVC "):
    #     svc = SVC(max_iter=100, tol=20, random_state=42)
    #     svc.fit(X, y)

    with Timer("Build and Fit: Linear Discriminant Analysis"):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)

    with Timer("Build and Fit: KNeighbors Classifier"):
        knn = KNeighborsClassifier()
        knn.fit(X, y)

    with Timer("Build and Fit: Decision Tree Classifier"):
        #dtc = DecisionTreeClassifier(class_weight=class_weights, random_state=42, ccp_alpha=0.01)
        dtc = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
        dtc.fit(X, y)

    with Timer("Build and Fit: Random Forest Classifier"):
        rfc = RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=3)
        rfc.fit(X, y)

        if save_models:
            with Timer("Saving models..."):
                _save_model(log_reg, "models\\" + dataset_used + "\\LogRegModel_Training.h5")
                _save_model(lda, "models\\" + dataset_used + "\\LDAModel_Training.h5")
                _save_model(knn, "models\\" + dataset_used + "\\KNNModel_Training.h5")
                _save_model(dtc, "models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
                _save_model(rfc, "models\\" + dataset_used + "\\RandomForestModel_Training.h5")
                # _save_model(svc, "models\\" + dataset_used + "\\SupportVectorClassifier_Training.h5")

    return {'Logistic_Regression': log_reg, 'LDA': lda, 'KNN': knn, 'Decision_Tree': dtc,
            'Random_Forest': rfc}


def build_ann_classifiers(X, y, class_weight, save_models=True, dataset_used="training"):
    X = Xs
    y = ys

    feature_size = X.shape[1]
    layer_size = 20
    n_epochs = 10
    batch_size = 64

    print(X.shape)
    exit()
    # input_layer =
    # active_layer = input_layer

    # active_layer = Conv2D(filters = 32,
    #                     kernel_size = (6,6),
    #                     activation=activation,
    #                     kernel_initializer='he_normal')(active_layer)

    # model4 = Sequential()
    # model4.add(Dense(layer_size, activation='relu', input_dim=feature_size))
    # model4.add(Dense(2, activation='softmax'))

    # model4.compile(optimizer='adam',
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #         metrics=['accuracy'])

    # model4.summary()

    # # setup logger so we can load history later
    # csv_logger = CSVLogger('training.log', separator=',', append=False)

    # history4 = model4.fit(x=X_train_rdy,
    #                         y=y_train,
    #                         validation_data=(X_val_rdy, y_val),
    #                         batch_size=batch_size,
    #                         epochs=n_epochs,
    #                         callbacks=[csv_logger])

    # print("Baseline model\n")

    # pred_train = model4.predict(X_train_rdy)
    # scores = model4.evaluate(X_train_rdy, y_train, verbose=0)
    # print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
    # print('\n')
    # pred_val= model4.predict(X_val_rdy)
    # scores2 = model4.evaluate(X_val_rdy, y_val, verbose=0)
    # print('Accuracy on validate data: {}% \n Error on validate data: {}'.format(scores2[1], 1 - scores2[1]))
    # print('\n')

    # # save model for later reference
    # #model4.save("step_5_initial_model_titanic")

    # truth = pd.DataFrame([[i] for i in y_train]).to_numpy()
    # pred = pred_train[:, 1]
    # pred[pred > 0.5] = 1
    # pred[pred <= 0.5] = 0

    # # print a classification report from sklearn
    # print(classification_report(y_true=truth, y_pred=pred, target_names=['died', 'survived']))

    # # plot the confusion matrix
    # plot_cm(truth, pred)

    # # baseline model
    # # baseline_ann =
    # # model = Sequential()
    # # model.add(Dense(60, input_shape=(60,), activation='relu'))
    # # model.add(Dense(1, activation='sigmoid'))
    # # # Compile model
    # # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    # # return model

    # # log_reg = LogisticRegression(solver='lbfgs', class_weight=class_weights, max_iter=500)
    # # log_reg.fit(X, y)

    # # lda = LinearDiscriminantAnalysis()
    # # lda.fit(X, y)

    # # qda = QuadraticDiscriminantAnalysis()
    # # qda.fit(X, y)

    # # knn = KNeighborsClassifier()
    # # knn.fit(X, y)

    # # dtc = DecisionTreeClassifier(class_weight=class_weights)
    # # dtc.fit(X, y)

    # # rfc = RandomForestClassifier(class_weight=class_weights)
    # # rfc.fit(X, y)

    # if save_models:
    #     # _save_model(log_reg, "models\\" + dataset_used + "\\LogRegModel_Training.h5")
    #     # _save_model(lda, "models\\" + dataset_used + "\\LDAModel_Training.h5")
    #     # _save_model(qda, "models\\" + dataset_used + "\\QDAModel_Training.h5")
    #     # _save_model(knn, "models\\" + dataset_used + "\\KNNModel_Training.h5")
    #     # _save_model(dtc, "models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
    #     # _save_model(rfc, "models\\" + dataset_used + "\\RandomForestModel_Training.h5")

    return {'Baseline': log_reg, 'LDA': lda, 'KNN': knn, 'Decision_Tree': dtc, 'Random_Forest': rfc}


def synch_datasets(df_training, df_test):
    for idx, col in enumerate(df_training):
        # print(f' training row: {col}, df_test.columns: {df_test.columns}')
        if col not in df_test.columns:
            df_training.drop(col, axis=1, inplace=True)
            print(f'Droppping {col} from training beacuse it is not in test data.')

    for idx, col in enumerate(df_test):
        # print(f' test row: {col}, df_training.columns: {df_training.columns}')
        if col not in df_training.columns:
            df_test.drop(col, axis=1, inplace=True)
            print(f'Droppping {col} from test beacuse it is not in training data.')

    return df_training, df_test


def predict_probs(models, X):
    """ Returns a dictionary of predicted proability vectors using models stored in the input dictionary 'models' on the feature data 'X'
    params:
    models - a dictionary of fitted classification models with key equal to the name of the model
    X - the values of a dataset obtained"""
    predicts = {}

    for key, model in models.items():
        predicts[key] = model.predict_proba(X)
    return predicts


def report_model_accuracy(models, Xs, ys):
    out_accuracy_str = '[ Accuracy report of classification models ]\n'

    X = Xs
    y = ys

    for i, (key, classifier) in enumerate(models.items()):
        # Get predictions
        y_pred = models[key].predict(X)
        # score = accuracy_score(y, pred)
        pred = cross_val_score(models[key], X, y, cv=5)
        mean_cv_accuracy = np.mean(pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        out_accuracy_str += key + " : " + str(mean_cv_accuracy) + "\n" + "bal acc:" + str(bal_acc)

    print("Finished report_model_accuracy")

    return out_accuracy_str


def _save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def _load_model(filename):
    return pickle.load(open(filename, 'rb'))


def load_all_models(dataset_used):
    with Timer("Load all models"):
        log_reg = _load_model("models\\" + dataset_used + "\\LogRegModel_Training.h5")
        lda = _load_model("models\\" + dataset_used + "\\LDAModel_Training.h5")
        knn = _load_model("models\\" + dataset_used + "\\KNNModel_Training.h5")
        dtc = _load_model("models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
        rfc = _load_model("models\\" + dataset_used + "\\RandomForestModel_Training.h5")
        # stk = _load_model("models\\" + dataset_used + "\\StackedEnsemble_Training.h5")
        brf = _load_model("models\\" + dataset_used + "\\BestRandomForest_Training.h5")

    return {'Logistic_Regression': log_reg, 'LDA': lda, 'KNN': knn, 'Random_Forest': rfc,
            'Best_Random_Forest': brf, 'Decision_Tree': dtc}


def evalutate_models(models, X, y):
    out_accuracy_str = '[ Accuracy report of classification models ]\n'

    X = Xs
    y = ys

    # estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)

    for i, (key, classifier) in enumerate(models.items()):
        # Get predictions
        pred = models[key].predict(X)
        score = accuracy_score(y, pred)

        out_accuracy_str += key + " : " + str(score) + "\n"

    return out_accuracy_str


# # evaluate model with standardized dataset
# estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def show_confusion_matrix(models, xs, ys):
    # Get our Xs and ys
    X = xs
    y = ys

    f, axes = plt.subplots(1, 6, figsize=(20, 5), sharey=True, sharex=True)

    for i, (key, classifier) in enumerate(models.items()):

        y_pred = classifier.predict(X)
        cf_matrix = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[i], xticks_rotation=45)
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    f.colorbar(disp.im_, ax=axes)
    plt.show()


def show_ROC_plots(models, xs, ys):
    # get true prediction results

    # check for nans
    if xs.isnull().values.any():
        print("found nulls in xs")
        exit()
    if ys.isnull().values.any():
        print("found nulls in ys")
        exit()
    y_true = ys
    test_predicts = predict_probs(models, xs)

    for i, (key, classifier) in enumerate(models.items()):
        RocCurveDisplay.from_predictions(y_true, test_predicts[key][:, 1], pos_label=1, ax=plt.gca(), name=key)

    # Add legend
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Add title
    plt.title("Receiver Operating Characteristic \n User trace classification")
    # print(str(df_names) + "\n Receiver Operating Characteristic")

    #plt.savefig('pml_figures/roc_curve.png')
    # Display plot
    plt.show()


# def analyze_models(models, ):


def get_stacked_model(Xs, ys, class_weights, dataset, models):
    X = Xs
    y = ys

    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', models['Logistic_Regression']),
            ('dtr', models['Decision_Tree']),
            ('brf', models['Best_Random_Forest']),
            # ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42)),
            # ('dt', DecisionTreeClassifier(class_weight=class_weights, random_state=42))
        ],
        final_estimator=RandomForestClassifier(class_weight=class_weights, random_state=43),
        cv=5  # number of cross-validation folds
    )
    stacking_clf.fit(X, y)
    _save_model(stacking_clf, "models\\" + dataset + "\\StackedEnsemble_Training.h5")

    pred = stacking_clf.predict(X)
    score = accuracy_score(y, pred)

    print("Stacked model accuracy score: " + str(score))

    return stacking_clf


def get_important_features_from_rand_forest(clf, Xs):
    # Feature importance:
    for score, name in zip(clf.feature_importances_, Xs.columns):
        print(round(score, 2), name)


def rand_grid_search_rand_forest(X, y, class_weights):
    with Timer("Hyperparameter Tuning - Random grid search"):
        print("Starting random grid search for Random Forest Classifier")
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=2, stop=24, num=12)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(2, 24, num=12)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier(class_weight=class_weights)
        # Random search of parameters, using 5 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                       random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(X, y)

        # get best model
        best_random = rf_random.best_estimator_
        # printaccuracy_rpt = report_model_accuracy({"Random Forest - Rand Grid Search":best_random}, X, y)

        # see how this model does
        _save_model(best_random, "models\\training\\BestRandomForest_Training.h5")

    # print best params:
    return best_random, rf_random.best_params_

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig(title+'.png')


def visualize_model(model,
                    title,
                    x_visualize: np.ndarray,
                    y_visualize: np.ndarray):
    """
    Visualize our predictions using classification report and confusion matrix

    :param model: the model used to make predictions for visualization
    :param x_visualize: the input features given used to generate prediction
    :param y_visualize: the true output to compare against the predictions
    """
    y_pred = model.predict(x_visualize)
    y_pred = np.array(y_pred > 0.5, dtype=int)
    #print("USING .6 THRESHOLD")
    #y_pred = np.array(y_pred > 0.6, dtype=int)
    y_true = y_visualize
    class_names = ['System', 'User']

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title='Confusion matrix, without normalization\n' + title)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix\n' + title)
    plt.show()


# Prune the tree!
def prune_decision_tree(X_train, y_train, X_test, y_test, class_weights):
    clf = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # plot to see best alpha
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    # now train the tree with the effective alphas
    clfs = []

    for ccp_alpha in ccp_alphas:
        if ccp_alpha < 0:
            clf = DecisionTreeClassifier(class_weight=class_weights, random_state=42, ccp_alpha=0)
            clf.fit(X_train, y_train)
            clfs.append(clf)
        else:
            clf = DecisionTreeClassifier(class_weight=class_weights, random_state=42, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    # Here we show that the number of nodes and tree depth decreases as alpha increases.
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

    # EXAMPLE
    # When ccp_alpha is set to zero and keeping the other default parameters of DecisionTreeClassifier,
    # the tree overfits, leading to a 100% training accuracy and 88% testing accuracy. As alpha increases,
    # more of the tree is pruned, thus creating a decision tree that generalizes better. In this example,

    # setting ccp_alpha=0.015 maximizes the testing accuracy.
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


    #_save_model(best_random, "models\\training\\BestRandomForest_Training.h5")


def grid_serach_cv_knn(X, y):
    # take actions and fit a c2_fixed_model that will do well on the test set
    k_range = list(range(1, 5))
    leaf_range = list(range(5, 20, 5))
    knn = KNeighborsClassifier(algorithm='auto')
    params = {
        'n_neighbors': k_range,
        'leaf_size': leaf_range,
        'p': (1, 2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev')
    }

    # with GridSearch
    grid_search_KNN = GridSearchCV(
        estimator=knn,
        param_grid=params,
        scoring='accuracy',
        n_jobs=-1,
        cv=5)

    # Commented out so the search does not run
    knn_grid = grid_search_KNN.fit(X, y)

    # Parameter setting that gave the best results on the hold out data.
    print(grid_search_KNN.best_params_)

    # Mean cross-validated score of the best_estimator
    print('Best Score - KNN:', grid_search_KNN.best_score_)


def create_decision_tree_graph(model, feature_names, title):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names=feature_names,
                    class_names=['0', '1'])

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_png(title+'_DecisionTree.png')
    Image(graph.create_png())

# This may not the best way to view each estimator as it is small
# def create_random_forest_tree_graph(rf_model, X, y)
#     fn=
#     cn="tag"
#     fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
#     for index in range(0, 4):
#         tree.plot_tree(rf_model.estimators_[index],
#                        feature_names = fn,
#                        class_names=cn,
#                        filled = True,
#                        ax = axes[index]);
#
#         axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
#     fig.savefig('rf_5trees.png')


def forward_selection_gridcv(model, X, y, cv):
    # X.drop("datetime", axis=1, inplace=True)

    # select only the numeric features to use
    numeric_features = X.select_dtypes('number').columns
    print(len(numeric_features), "numeric features:", numeric_features)

    selected_features_forward = []  # placeholder to contain list of feature name text strings (column names)
    model_scores_forward = []  # placeholder to keep list of scores per fitted model (length = feature qty-1)
    best_score_forward = np.inf  # start as bad as possible, to be replaced with best score
    best_idx_forward = None  # placeholder
    best_features_forward = None  # placeholder to contain list of feature name text strings (column names)

    for idx, num_feats in enumerate(range(1, len(numeric_features))):
        with Timer():
            # ----------------START STUDENT CODE -----------------------
            print(f'Starting {num_feats} features')

            # Create Sequential Feature Selector with Cross validation, dataspace_rmse for each number of features
            sfs_fwd = SequentialFeatureSelector(model, cv=cv, scoring='balanced_accuracy',
                                                n_features_to_select=num_feats)

            # Fit the Selector with numeric features.
            sfs_fwd.fit(X[numeric_features], y)

            # Store features selected during each iteration for reference
            selected_features_forward.append(sfs_fwd.get_support(indices=False))

            # Set X for the selected features of this number of features best selection
            X_sfs_fwd = X[numeric_features].iloc[:, selected_features_forward[idx]]

            # Fit the model using the selected features
            model.fit(X_sfs_fwd, y)

            # Score using cv, make it postive and take the mean as well
            sfs_fwd_score = np.mean(np.abs(
                cross_val_score(model, X_sfs_fwd, y,
                                scoring='balanced_accuracy',
                                cv=cv,
                                n_jobs=-1)))

            # Store score for later reference
            model_scores_forward.append(sfs_fwd_score)

            # Update best score if the current
            if sfs_fwd_score < best_score_forward:
                best_score_forward = sfs_fwd_score
                best_idx_forward = idx
                best_features_forward = selected_features_forward[idx]

        print(f'Completed {num_feats} features')

    print(f'Best greedy forward: {best_score_forward}, with features: {best_features_forward}')
    print(X.columns.values.tolist())
    print("\n With parameters: \n\n", X[numeric_features].iloc[:, selected_features_forward[best_idx_forward]])

    out = X[numeric_features].iloc[:, selected_features_forward[best_idx_forward]]
    out.to_csv("best_features_forward_selection.csv")


def grid_serach_cv_logistic_regression(X, y):
    #parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
     #             {'penalty':['none', 'elasticnet', 'l1', 'l2']},
    #              {'C':[0.001, 0.01, 0.1, 1, 10, 100]}]
    grid = {"C": np.logspace(-3, 3, 100), "penalty": ["l1", "l2", "none", 'elasticnet']}  # l1 lasso l2 ridge
    logreg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1500)
    logreg_cv = GridSearchCV(logreg, grid, cv=5)
    logreg_cv.fit(X, y)

    _save_model(logreg_cv, "models\\training\\LogRegModel_Training.h5")

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)


def precision_recall_threshold(model, X, y):

    test_x = X
    test_y = y

    pred_y=model.predict(test_x)
    probs_y=model.predict_proba(test_x)

    precision, recall, thresholds = precision_recall_curve(test_y, probs_y[:, 1])
    #retrieve probability of being 1(in second column of probs_y)
    pr_auc = metrics.auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    #plt.savefig("pml_figures/precision_recall_curve.png")
    plt.show()

def custom_predict(log_model, X, threshold):
    probs = log_model.predict_proba(X)
    return (probs[:, 1] > threshold).astype(int)


# plot the confusion matrix 
def plot_cm(actual: np.ndarray, prediction: np.ndarray):
    """
    make a plot for a confusion matrix
    
    :param actual: the actual classes 
    :param prediction:  the predicted classes
    :return: None
    """
    # use the probabilities to make actual predictions of each class
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    # use pandas to make a confusion matrix
    data = {'y_Actual': actual.squeeze(),
            'y_Predicted': prediction.squeeze()
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    # use seaborn to plot a heatmap of the confusion matrix
    sns.heatmap(confusion_matrix, annot=True)
    #plt.show()
    plt.savefig('confusion_matrix.png')
