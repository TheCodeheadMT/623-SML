import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, RocCurveDisplay, confusion_matrix
import SMLFinalPreprocessor as util


#### START: PREPROCESS DATA ################################################
PREPROCESS = False
if PREPROCESS:
    CUR_SET = "test"

    # read in dataset1.csv
    if CUR_SET == "training":
        df = pd.read_csv("MANUAL_LABELED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK_training_timeline1.csv", header=0,
                         names=['datetime', 'timestamp_desc', 'source', 'source_long',
                                'message', 'parser', 'display_name', 'tag'])
    else:
        df = pd.read_csv("MANUAL_LABELED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK_test_timeline1.csv", header=0,
                        names=['datetime', 'timestamp_desc', 'source', 'source_long',
                                'message', 'parser', 'display_name', 'tag'])

    # preprocess the timestamp field
    df.datetime = util.clean_datetimes(df.iloc[:, 0])

    # TESTING
    df = util.encode_file_depth_feature(df)

    # add feature: first directory in path
    # df = util.parse_first_dir_from_display_name_feature(df)

    # add feature: msie cache message length
    df = util.get_msg_len_feature(df)

    # add feature: distance from flagged user activity
    #df = util.get_time_delta_from_tagged_activity_feature(df)

    # add feature: endcoded display feature
    df = util.encode_display_name_feature(df)

    #Significant features: tag, file_depth, msg_len, flagged_activity_dist, flagged_activity_dist_X_file_depth
    # create interaction feature between flagged_activity_dist * file_depth
    # this works beacuse files were flagged as releavnt by the LCS, then artifacts near flagged objects
    # in the same depth for the display are closer in distance
    #df['flagged_activity_dist_X_file_depth'] = df['flagged_activity_dist'] * df['file_depth']

    # save to inspect before encoding
    df.to_csv("raw_"+CUR_SET+".csv", index=False)
    #df.to_csv("Test_before_encoding_test.csv", index=False)

    # scale and encode the dataset
    df_transformed = util.transform_win7_traces(df)

    if CUR_SET == 'training':
        df_transformed.to_csv("MANUAL_LABELED_ENCODED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-training_timeline1.txt",index=False)
    else:
        df_transformed.to_csv("MANUAL_LABELED_ENCODED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-test_timeline1.txt", index=False)
    exit()
    ####   END: PREPROCESS DATA ################################################

############################################################################

NORMAL_CLS = True
if NORMAL_CLS:

    ####### DESCRIBE DATA ##########################################################
    DESCRIBE_DATA = False
    if DESCRIBE_DATA:
        pass
        # Read in training data
        df1 = pd.read_csv("raw_training.csv")

        # select only the columns of interest
        df_new = df1[['tag', 'file_depth', 'msg_len']]
        le = preprocessing.LabelEncoder()
        le.fit(df1['source'])
        df_new['source'] = le.transform(df1['source']) + 1

        le = preprocessing.LabelEncoder()
        le.fit(df1['source_long'])
        df_new['source_long'] = le.transform(df1['source_long']) + 1

        le = preprocessing.LabelEncoder()
        le.fit(df1['key_dirs'])
        df_new['key_dirs'] = le.transform(df1['key_dirs']) + 1

        util.corr_plots(df_new, title="Raw Training dataset Correlation")

        util.data_explore_msg_len_file_depth(df_new[['tag', 'file_depth', 'msg_len']])

        util.box_plots(df_new)

        # pairwise plots
        #f = plt.figure()
        #f.set_title("File_depth and Msg_len Pairs")
        #df1['msg_len'] = np.log2(df1['msg_len'])
        #df1['file_depth'] = np.log2(df1['file_depth'])
        #sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        #g = sns.PairGrid(df1, hue="tag")

        #long running, so commented out, dont delete
        # fig = sns.pairplot(df_new, hue='tag')
        # fig.fig.suptitle("Pairs Plot", y=1.08) # y= some height>1
        # fig.add_legend()
        # plt.show()
        exit()




    ####### LOAD DATA ##########################################################
    # read in dataset1.csv
    df1 = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-training_timeline1.txt")

    df2 = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-test_timeline1.txt")


    # remove any columns that are not in both datasets
    util.synch_datasets(df1, df2)

    ####### SPLIT DATA FOR PROCESSING ###############################################

    # Split data into two equal groups for training and validation maintaining class distributions
    #df1_train, df1_validation = train_test_split(df1, train_size=0.7, random_state=42, stratify=df1['tag'])
    df1_train = df1
    df2_test = df2

    # Set weights for training
    n_class_1 = df1_train[df1_train['tag'] == 1].shape[0]
    n_class_0 = df1_train.shape[0] - n_class_1
    n_samples = df1_train.shape[0]

    # Set class weights for training
    class_weight = {
        0: n_samples / (2 * n_class_0),
        1: n_samples / (2 * n_class_1)
    }

    # Set Xs and Ys for our training set
    TRAIN = True
    if TRAIN:
        X = df1_train.drop(['tag', 'datetime'], axis=1)
        y = df1_train.tag
        X_test = df2_test.drop(['tag', 'datetime'], axis=1)
        y_test = df2_test.tag

        # Get feature names
        feature_names = X.columns.values.tolist()

        # Build or load models
        LOAD_MODELS = True
        if LOAD_MODELS:
                df1_train_models = util.load_all_models("training")
                #util.grid_search_decision_tree(X,y, class_weight)

        else:
            print("Training models...")
            df1_train_models = util.build_classifiers(X, y, class_weight, save_models=True, dataset_used="training")


            # Create a stacked model and then report accuracy for all
            #stked_clf = util.get_stacked_model(X, y, class_weight, "training")
            #df1_train_models['Stacking_Ensemble'] = stked_clf
            #mod_, params_ = util.rand_grid_search_rand_forest(X, y, class_weight)
            #df1_train_models['Best_Random_Forest'] = mod_

            #util.prune_decision_tree(X, y, X_test, y_test, class_weight)


            print("All models trained ...\n")
            exit()
        # check for important features using Random Forest Classifier
        #print(util.get_important_features_from_rand_forest(df1_train_models['Random_Forest'], X))
        #print(util.get_important_features_from_rand_forest(df1_train_models['Best_Random_Forest'], X))

        FEATURE_SELECTION = False
        if FEATURE_SELECTION:
            util.forward_selection_gridcv(df1_train_models['Logistic_Regression'], X, y, cv=2)
            exit()

        TUNE_HYPERPARAMS = False
        if TUNE_HYPERPARAMS:
            #stacked_mod = util.get_stacked_model(X, y, class_weight, "training", df1_train_models)
            #df1_train_models['Stacking_Model'] = stacked_mod

            util.grid_serach_cv_logistic_regression(X,y)
            exit()

            #mod_, params_ = util.rand_grid_search_rand_forest(X, y, class_weight)
            #df1_train_models['Best_Random_Forest'] = mod_


        REPORT_RESULTS = True
        if REPORT_RESULTS:
            df1_train_models = {"Logistic_Regression": df1_train_models['Logistic_Regression'],
                                "Decision_Tree": df1_train_models['Decision_Tree']}

            util.plot_feature_importance(df1_train_models['Decision_Tree'], X, y)
            exit()


            #print(df1_train_models['KNN'])
            # for parameter in df1_train_models['KNN'].get_params():
            #     print(parameter, ': \t', df1_train_models['KNN'].get_params()[parameter])


            # Print accuracy results
            print(util.report_model_accuracy(df1_train_models, X, y, thresholds=[0.30, 0.50]))
            #util.visualize_model(df1_train_models['Logistic_Regression'], 'Logistic Regression\n C=0.001, l2 penalty', X, y )
            #util.visualize_model(df1_train_models['Decision_Tree'], 'Decision Tree \n CCP Alpha=0.01', X, y)

            # Show confusion matrix for all models
            util.show_confusion_matrix(df1_train_models, X, y)

            # Show ROC curves
            util.show_ROC_plots(df1_train_models, X, y)

            util.precision_recall_threshold(df1_train_models['Logistic_Regression'], X, y, "\n Logistic Regression")
            util.precision_recall_threshold(df1_train_models['Decision_Tree'], X, y, "\n Decision Tree")

            util.visualize_model(df1_train_models['Logistic_Regression'], "Logistic Regression (.53)", X, y, threshold=0.30)
            util.visualize_model(df1_train_models['Decision_Tree'], "Decision Tree (.50)", X, y, threshold=0.50)




            #predicts = pd.DataFrame(df1_train_models['Logistic_Regression'].predict_proba(X))
            #predicts.to_csv("training_predictions_LR.csv")
            #predicts = pd.DataFrame(df1_train_models['Decision_Tree'].predict_proba(X))
            #predicts.to_csv("training_predictions_DT.csv")



            #Create decision tree visualization
            #util.create_decision_tree_graph(df1_train_models['Decision_Tree'], feature_names)
            # for parameter in df1_train_models['Best_Random_Forest'].get_params():
            #     print(parameter, ': \t', df1_train_models['Best_Random_Forest'].get_params()[parameter])

            # for idx in range(0, 4):
            #     util.create_decision_tree_graph(df1_train_models['Best_Random_Forest'].estimators_[idx],
            #                                     feature_names,
            #                                     'Estimator_'+str(idx))

            # for idx, (key, md) in enumerate(df1_train_models.items()):
            #     util.visualize_model(md, key, X, y)


            #util.prune_decision_tree(X, y, X_test, y_test, class_weight)
            exit()

        TEST_WITH_BEST_MODEL = True
        if TEST_WITH_BEST_MODEL:
            print('Testing best model:')

            # Test the best model
            #best_model = df1_train_models['Logistic_Regression']
            df1_train_models = {"Logistic_Regression": df1_train_models['Logistic_Regression'],
                                "Decision_Tree": df1_train_models['Decision_Tree']}


            #util.precision_recall_threshold(df1_train_models['Logistic_Regression'], X_test, y_test)
            #util.precision_recall_threshold(df1_train_models['Decision_Tree'], X_test, y_test)

            util.visualize_model(df1_train_models['Logistic_Regression'], "Logistic Regression (.53)", X_test, y_test, threshold=0.30)
            util.visualize_model(df1_train_models['Decision_Tree'], "Decision Tree", X_test, y_test, threshold=0.5)

            #util.create_decision_tree_graph(df1_train_models['Decision_Tree'], feature_names, "Decision Tree Graph (alpha-0.001 TH-0.60)")

        #best_model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, C=0.001, penalty='l2')
            #log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
            #best_model.fit(X, y)
            #best_model = df1_train_models['Logistic_Regression']
            print(util.report_model_accuracy(df1_train_models, X_test, y_test, thresholds=[0.30, 0.50]))



        # print prarms:
            for parameter in df1_train_models['Decision_Tree'].get_params():
                print(parameter, ': \t', df1_train_models['Decision_Tree'].get_params()[parameter])

            #pred = best_model.predict(X_test)
            #score = accuracy_score(y_test, pred)

            # check performance on new data...
            #util.visualize_model(best_model, 'Logistic_Regression', X_test, y_test )

            #for idx, (key, md) in enumerate(df1_train_models.items()):
            #    util.visualize_model(md, key, X_test, y_test)

            util.show_confusion_matrix(df1_train_models, X_test, y_test)

            util.show_ROC_plots(df1_train_models, X_test, y_test)

            #util.precision_recall_threshold(df1_train_models, X_test,y_test)


    # test against our test data
    # util.analyze_models(models)


