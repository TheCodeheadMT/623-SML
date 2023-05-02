import datetime as datetime
import pickle
from statistics import mean

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
import xgboost as xgb

dont_explore = ['timestamp_desc_Added Time', 'timestamp_desc_Content Deletion Time', 'timestamp_desc_End Time',
                'timestamp_desc_Expiration Time', 'timestamp_desc_File Last Modification Time',
                'timestamp_desc_Installation Time', 'timestamp_desc_Last Access Time',
                'timestamp_desc_Last Connection Time', 'timestamp_desc_Last Login Time',
                'timestamp_desc_Last Password Set Time',
                'timestamp_desc_Last Shutdown Time', 'timestamp_desc_Last Time Executed',
                'timestamp_desc_Last Visited Time', 'timestamp_desc_Last registered Time', 'timestamp_desc_Launch Time',
                'timestamp_desc_Metadata Modification Time', 'timestamp_desc_Synchronization Time',
                'timestamp_desc_Unknown Time', 'source_EVT', 'source_LOG', 'source_OLECF', 'source_PE', 'source_RECBIN',
                'source_REG', 'source_WEBHIST', 'source_long_AppCompatCache Registry Key',
                'source_long_BagMRU Registry Key', 'source_long_Chrome Cache', 'source_long_Firefox History',
                'source_long_MRUList Registry Key',
                'source_long_MRUListEx Registry Key', 'source_long_MSIE Cache File URL record',
                'source_long_Microsoft Office MRU Registry Key', 'source_long_NTFS USN change',
                'source_long_OLECF Dest list entry',
                'source_long_OLECF Item', 'source_long_PE/COFF file', 'source_long_Recycle Bin',
                'source_long_Registry Key', 'source_long_Run/Run Once Registry Key',
                'source_long_Service/Driver Configuration Registry Key',
                'source_long_Setup API Log', 'source_long_Shutdown Registry Key', 'source_long_System',
                'source_long_Task Cache Registry Key', 'source_long_Typed URLs Registry Key',
                'source_long_USB Registry Key',
                'source_long_USBStor Registry Key', 'source_long_User Account Information Registry Key',
                'source_long_UserAssist Registry Key', 'source_long_WinEVTX', 'source_long_WinPrefetch',
                'source_long_Winlogon Registry Key']


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


def data_explore(df):
    df_focused = df.drop(dont_explore, axis=1)

    for ftr in df_focused:
        if ftr not in dont_explore:
            # print description of this feature
            print(ftr, " description:\n", df_focused[ftr].describe(), '\n')

            # pairwise plots
            sns.pairplot(df_focused, hue='tag')

            # show covariance for class 0
            print("Covariance Class 0:\n", df_focused[df_focused.tag == 0].cov(), '\n')

            # show a historgram for each class 0, 1
            df_focused[df_focused.class_tag == 0].hist(figsize=(20, 15), legend=True, bins=50)
            plt.title("Class 0 Histogram Plots")
            plt.show()

            # show covariance for class 1
            print("Covariance Class 1:\n", df_focused[df_focused.tag == 1].cov(), '\n')

            df_focused[df_focused.class_tag == 1].hist(figsize=(20, 15), legend=True, bins=50)
            plt.title("Class 1 Histogram Plots")
            plt.show()


def box_plots(df):
    # reverse one-hot encoding
    # df[] = (df.iloc[:, 2:] == 1).idxmax(1)

    # actuals and predicted to one column
    # df = df.melt(id_vars ='season', value_vars=['actuals', 'predicted'])

    # sn.boxplot(x='season', y='value', hue='variable', data=df, palette="Set3")
    return df


def transform_win7_traces(features_df):
    features_out = features_df
    tags = features_df.tag

    # features not used at all in model
    primary_exclude_list = ['message', 'display_name', 'parser', 'tag']

    # nominal features used in model
    # nom_features_list = ['timestamp_desc', 'source', 'source_long']
    nom_features_list = ['source', 'source_long', 'timestamp_desc']

    # numeric features used in model
    num_features_list = ['datetime']

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

    return df_out


def build_classifiers(Xs, ys, class_weights, save_models=False, dataset_used="training"):
    X = Xs
    y = ys

    log_reg = LogisticRegression(solver='lbfgs', class_weight=class_weights, max_iter=500)
    log_reg.fit(X, y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    dtc = DecisionTreeClassifier(class_weight=class_weights)
    dtc.fit(X, y)

    rfc = RandomForestClassifier(class_weight=class_weights)
    rfc.fit(X, y)

    if save_models:
        _save_model(log_reg, "models\\" + dataset_used + "\\LogRegModel_Training.h5")
        _save_model(lda, "models\\" + dataset_used + "\\LDAModel_Training.h5")
        _save_model(qda, "models\\" + dataset_used + "\\QDAModel_Training.h5")
        _save_model(dtc, "models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
        _save_model(rfc, "models\\" + dataset_used + "\\RandomForestModel_Training.h5")

    return {'Logistic_Regression': log_reg, 'LDA': lda, 'QDA': qda, 'Decision_Tree': dtc, 'Random_Forest': rfc}

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
    # create table to populate
    df_out = df = pd.DataFrame({
        'Dataset': [],
        'Logistic Regression': [],
        'Linear Discriminant Analysis': [],
        'Quadradic Discriminant Analysis': [],
        'Decision Tree Classifier': [],
        'Random Forest Classifier': []
    })

    # print(ds)
    # Get our Xs and ys
    X = Xs
    y = ys

    # Get predictions
    lg_pred = models['Logistic_Regression'].predict(X)
    # Get accuracy score
    lg_score = accuracy_score(y, lg_pred)

    # Get predictions
    lda_pred = models['LDA'].predict(X)
    # Get accuracy score
    lda_score = accuracy_score(y, lda_pred)

    # get predictions
    qda_pred = models['QDA'].predict(X)
    qda_score = accuracy_score(y, qda_pred)

    # predict responses
    dtc_pred = models['Decision_Tree'].predict(X)
    dtc_score = accuracy_score(y, dtc_pred)

    # predict responses
    rfc_pred = models['Random_Forest'].predict(X)
    rfc_score = accuracy_score(y, rfc_pred)

    # add the row of scores to our data frame
    df_out.loc[0] = ["Dataset", lg_score, lda_score, qda_score, dtc_score, rfc_score]

    return df_out


def _save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def _load_model(filename):
    return pickle.load(open(filename, 'rb'))


def load_all_models(dataset_used):
    log_reg = _load_model("models\\" + dataset_used + "\\LogRegModel_Training.h5")
    lda = _load_model("models\\" + dataset_used + "\\LDAModel_Training.h5")
    qda = _load_model("models\\" + dataset_used + "\\QDAModel_Training.h5")
    dtc = _load_model("models\\" + dataset_used + "\\DecisionTreeModel_Training.h5")
    rfc = _load_model("models\\" + dataset_used + "\\RandomForestModel_Training.h5")

    return {'Logistic_Regression': log_reg, 'LDA': lda, 'QDA': qda, 'Decision_Tree': dtc, 'Random_Forest': rfc}


def show_confusion_matrix(models, xs, ys):
    # Get our Xs and ys
    X = xs
    y = ys

    f, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True, sharex=True)

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
    #get true prediction results
    y_true = ys
    test_predicts = predict_probs(models, xs)

    # Log Rec ROC
    RocCurveDisplay.from_predictions(y_true, test_predicts['Logistic_Regression'][:, 1], pos_label=1, ax=plt.gca(), name="Log Reg")

    # LDA ROC
    RocCurveDisplay.from_predictions(y_true,  test_predicts['LDA'][:, 1], pos_label=1, ax=plt.gca(), name="LDA")

    # QDA ROC
    RocCurveDisplay.from_predictions(y_true, test_predicts['QDA'][:, 1], pos_label=1, ax=plt.gca(), name="QDA")

    # LDA ROC
    RocCurveDisplay.from_predictions(y_true,  test_predicts['Decision_Tree'][:, 1], pos_label=1, ax=plt.gca(), name="Decision Tree")

    # QDA ROC
    RocCurveDisplay.from_predictions(y_true, test_predicts['Random_Forest'][:, 1], pos_label=1, ax=plt.gca(), name="Random Forest")

    # Add legend
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Add title
    plt.title("Receiver Operating Characteristic \n User trace classification")
    #print(str(df_names) + "\n Receiver Operating Characteristic")

    # Display plot
    plt.show()

def gridSearchCV(model, validation_data):
    X = validation_data.drop(["datetime", "tag"], axis=1)
    y = validation_data.tag

    model = model['Decision_Tree']

    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # summarize performance
    print('Mean ROC AUC: %.3f' % mean(scores))

    #
    # # setup model to be tuned
    # #model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    # # put weights in a dict
    # param_grid = dict(scale_pos_weights=weights)
    #
    # # define evalu prodcedure
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    # # define grid search
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # # execute search
    # grid.fit(X, y)
    # print("Best %f using %s" % (grid.best_score_, grid.best_params_))
    # # report configs
    # means = grid.cv_results_['mean_test_score']
    # stds = grid.cv_results_['std_test_score']
    # params = grid.cv_results_['params']
    # for mean, std, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" %  (mean, std, param))
