import datetime as datetime
import itertools
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from keras import Model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
# from keras.models import KerasClassifier
# from sklearn.keras.wrappers.scikit_learn.KerasClassifier,
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
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

def corr_plots(df):
    dataplot = sns.heatmap(df.corr().sort_values('tag', ascending=False), cmap='YlGnBu')
    print(df.corr())
    plt.show()

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
                out_str = out_str + dir+"\\"
            else:
                out_str = out_str + "\\*"
    return out_str


# transform display_name column into simplified represenatation
def encode_display_name_feature(df):
    with Timer("Add feature - encoded display name"):
        df['key_dirs'] = df['display_name'].apply(encode_display_name)
    return df

# def data_explore(df):

# df_focused = df.drop(dont_explore, axis=1)

# for ftr in df_focused:
#     if ftr not in dont_explore:
#         # print description of this feature
#         print(ftr, " description:\n", df_focused[ftr].describe(), '\n')
#
#         # pairwise plots
#         sns.pairplot(df_focused, hue='tag')
#
#         # show covariance for class 0
#         print("Covariance Class 0:\n", df_focused[df_focused.tag == 0].cov(), '\n')
#
#         # show a historgram for each class 0, 1
#         df_focused[df_focused.class_tag == 0].hist(figsize=(20, 15), legend=True, bins=50)
#         plt.title("Class 0 Histogram Plots")
#         plt.show()
#
#         # show covariance for class 1
#         print("Covariance Class 1:\n", df_focused[df_focused.tag == 1].cov(), '\n')
#
#         df_focused[df_focused.class_tag == 1].hist(figsize=(20, 15), legend=True, bins=50)
#         plt.title("Class 1 Histogram Plots")
#         plt.show()


def box_plots(df):
    # reverse one-hot encoding
    # df[] = (df.iloc[:, 2:] == 1).idxmax(1)

    # actuals and predicted to one column
    # df = df.melt(id_vars ='season', value_vars=['actuals', 'predicted'])

    # sn.boxplot(x='season', y='value', hue='variable', data=df, palette="Set3")
    return df


def transform_win7_traces(features_df):
    with Timer("Transform Win 7 Traces"):
        features_out = features_df
        tags = features_df.tag

        # features not used at all in model
        primary_exclude_list = ['message', 'display_name', 'parser', 'tag']

        # nominal features used in model
        # nom_features_list = ['timestamp_desc', 'source', 'source_long']
        nom_features_list = ['source', 'source_long', 'timestamp_desc', "key_dirs"]
        # 'first_dir'

        # numeric features used in model
        num_features_list = ['datetime', 'msg_len']

        # Removed 'flagged_activity_dist'

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
        df_out = df_out.drop("source_long_Chrome Cache", axis=1)

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
                print("-> % forward complete: ", index / df.shape[0])

    return df


def build_classifiers(Xs, ys, class_weights, save_models=False, dataset_used="training"):
    X = Xs
    y = ys

    with Timer("Build and Fit: Logistic Regression"):
        log_reg = LogisticRegression(solver='lbfgs', class_weight=class_weights, max_iter=500)
        log_reg.fit(X, y)

    with Timer("Build and Fit: Linear Discriminant Analysis"):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
    with Timer("Build and Fit: KNeighbors Classifier"):
        knn = KNeighborsClassifier()
        knn.fit(X, y)
    with Timer("Build and Fit: Decision Tree Classifier"):
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
        pred = models[key].predict(X)
        score = accuracy_score(y, pred)

        out_accuracy_str += key + " : " + str(score) + "\n"

    print("finished report_model_accuracy")

    return out_accuracy_str


def _save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def _load_model(filename):
    return pickle.load(open(filename, 'rb'))


def load_all_models(dataset_used):
    with Timer("Load all models"):
        log_reg = _load_model("models\\" + dataset_used + "\\LogRegModel_Training.h5")
        lda = _load_model("models\\" + dataset_used + "\\LDAModel_Training.h5")
        # qda = _load_model("models\\" + dataset_used + "\\QDAModel_Training.h5")
        knn = _load_model("models\\" + dataset_used + "\\KNNModel_Training.h5")
        dtc = _load_model("models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
        rfc = _load_model("models\\" + dataset_used + "\\RandomForestModel_Training.h5")
        stk = _load_model("models\\" + dataset_used + "\\StackedEnsemble_Training.h5")

    return {'Logistic_Regression': log_reg, 'LDA': lda, 'KNN': knn, 'Decision_Tree': dtc,
            'Random_Forest': rfc, 'Stacking_Ensemble': stk}


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

    # Display plot
    plt.show()


# def analyze_models(models, ):


def get_stacked_model(Xs, ys, class_weights, dataset):
    X = Xs
    y = ys

    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(solver='lbfgs', class_weight=class_weights, random_state=42, max_iter=500)),
            ('ldr', LinearDiscriminantAnalysis()),
            # ('knn', KNeighborsClassifier()),
            # ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42)),
            # ('dt', DecisionTreeClassifier(class_weight=class_weights, random_state=42))
        ],
        # final_estimator=RandomForestClassifier(class_weight=class_weights, random_state=43),
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


# def gridSearchCV(model, data):
#     X = data.drop(["datetime", "tag"], axis=1)
#     y = data.tag
#     #    minLogAlpha = -3
#     #    maxLogAlpha = 7
#     #    alphaCount = 1000
#     #    alphagrid = np.zeros(alphaCount)  # placeholder for the alphas
#     #    alphagrid = np.logspace(minLogAlpha,maxLogAlpha,num=alphaCount)
#
#     model1 = model['Decision_Tree']
#     tree_param = {'criterion': ['gini', 'entropy'],
#                   'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
#     lcv_model = None  # placeholder for GridSearchCV() wrapper of Lasso() model
#     best_lasso_alpha_decision_tree = None  # placeholder
#     best_lasso_score_decision_tree = None  # placeholder
#
#     model2 = model['Random_Forest']
#     lcv_model = None  # placeholder for GridSearchCV() wrapper of Lasso() model
#     best_lasso_alpha_decision_tree = None  # placeholder
#     best_lasso_score_decision_tree = None  # placeholder
#
#     # # define evaluation procedure
#     # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     # # evaluate model
#     # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#     # # summarize performance
#     # print('Mean ROC AUC: %.3f' % mean(scores))
#
#     lasso_cv_results = pd.DataFrame()  # placeholder
#
#     # ------------- START STUDENT CODE ------------------
#
#     # Create Lasso model
#     lasso = Lasso()
#
#     # Wrap Lasso() GridSearch CV using Lasso, alphagrid, dataspace_rmse scoring, kfold_count and return training scores.
#     lcv_model = GridSearchCV(Lasso(),
#                              {'alpha': alphagrid},
#                              scoring=dataspace_rmse,
#                              cv=kfold_count,
#                              return_train_score=True)
#
#     # Fit GridSearchCV to nonTest numeric features and log_y_nontest
#     lcv_model.fit(X_nonTest_scaled[numeric_features], log_y_nonTest)
#
#     # Store best lasso alpha
#     best_lasso_alpha = lcv_model.best_params_['alpha']
#
#     # Store best lasso score - multiplied by -1 for comparision with other scores
#     best_lasso_score = lcv_model.best_score_ * -1
#
#     # Store results in dataframe for later reference
#     lasso_cv_results = pd.DataFrame(lcv_model.cv_results_)
#
#     #
#     # # setup model to be tuned
#     # #model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
#     # # put weights in a dict
#     # param_grid = dict(scale_pos_weights=weights)
#     #
#     # # define evalu prodcedure
#     # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
#     # # define grid search
#     # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
#     # # execute search
#     # grid.fit(X, y)
#     # print("Best %f using %s" % (grid.best_score_, grid.best_params_))
#     # # report configs
#     # means = grid.cv_results_['mean_test_score']
#     # stds = grid.cv_results_['std_test_score']
#     # params = grid.cv_results_['params']
#     # for mean, std, param in zip(means, stds, params):
#     #     print("%f (%f) with: %r" %  (mean, std, param))

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



