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

        util.data_explore(df_new[['tag', 'file_depth', 'msg_len']])

        # pairwise plots
        #f = plt.figure()
        #f.set_title("File_depth and Msg_len Pairs")
        #df1['msg_len'] = np.log2(df1['msg_len'])
        #df1['file_depth'] = np.log2(df1['file_depth'])
        #sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})
        #g = sns.PairGrid(df1, hue="tag")
        fig = sns.pairplot(df_new, hue='tag')
        fig.fig.suptitle("Pairs Plot", y=1.08) # y= some height>1
        fig.add_legend()
        plt.show()
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
        LOAD_MODELS = False
        if LOAD_MODELS:
                df1_train_models = util.load_all_models("training")
        else:
            print("Training models...")
            df1_train_models = util.build_classifiers(X, y, class_weight, save_models=True, dataset_used="training")
            # Create a stacked model and then report accuracy for all
            #stked_clf = util.get_stacked_model(X, y, class_weight, "training")
            #df1_train_models['Stacking_Ensemble'] = stked_clf
            #mod_, params_ = util.rand_grid_search_rand_forest(X, y, class_weight)
            #df1_train_models['Best_Random_Forest'] = mod_
            print("All models trained ...\n")

        # check for important features using Random Forest Classifier
        #print(util.get_important_features_from_rand_forest(df1_train_models['Random_Forest'], X))
        #print(util.get_important_features_from_rand_forest(df1_train_models['Best_Random_Forest'], X))

        FEATURE_SELECTION = False
        if FEATURE_SELECTION:
            util.forward_selection_gridcv(df1_train_models['Logistic_Regression'], X, y, cv=2)
            exit()

        TUNE_HYPERPARAMS = True
        if TUNE_HYPERPARAMS:
            #stacked_mod = util.get_stacked_model(X, y, class_weight, "training", df1_train_models)
            #df1_train_models['Stacking_Model'] = stacked_mod

            util.grid_serach_cv_logistic_regression(X,y)
            exit()

            #mod_, params_ = util.rand_grid_search_rand_forest(X, y, class_weight)
            #df1_train_models['Best_Random_Forest'] = mod_


        REPORT_RESULTS = True
        if REPORT_RESULTS:
            df1_train_models = {"Logistic_Regression":df1_train_models['Logistic_Regression'],
                                "Decision_Tree":df1_train_models['Decision_Tree']}

            # Print accuracy results
            print(util.report_model_accuracy(df1_train_models, X, y))

            util.visualize_model(df1_train_models['Logistic_Regression'], 'Logistic Regression\n C=0.001, l2 penalty', X, y )
            util.visualize_model(df1_train_models['Decision_Tree'], 'Decision Tree \n CCP Alpha=0.01', X, y)

            # Show confusion matrix for all models
            util.show_confusion_matrix(df1_train_models, X, y)

            # Show ROC curves
            util.show_ROC_plots(df1_train_models, X, y)

            #Create decision tree visualization
            #util.create_decision_tree_graph(df1_train_models['Decision_Tree'], feature_names)
            # for parameter in df1_train_models['Best_Random_Forest'].get_params():
            #     print(parameter, ': \t', df1_train_models['Best_Random_Forest'].get_params()[parameter])

            # for idx in range(0, 4):
            #     util.create_decision_tree_graph(df1_train_models['Best_Random_Forest'].estimators_[idx],
            #                                     feature_names,
            #                                     'Estimator_'+str(idx))

            for idx, (key, md) in enumerate(df1_train_models.items()):
                util.visualize_model(md, key, X, y)


            #util.prune_decision_tree(X, y, X_test, y_test, class_weight)


        TEST_WITH_BEST_MODEL = True
        if TEST_WITH_BEST_MODEL:
            print('Testing best model:')

            # Test the best model
            best_model = df1_train_models['Logistic_Regression']
            #best_model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, C=0.001, penalty='l2')
            #log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
            #best_model.fit(X, y)
            #best_model = df1_train_models['Logistic_Regression']


        # print prarms:
            for parameter in best_model.get_params():
                print(parameter, ': \t', best_model.get_params()[parameter])

            #pred = best_model.predict(X_test)
            #score = accuracy_score(y_test, pred)

            #

            # check performance on new data...
            util.visualize_model(best_model, 'Logistic_Regression', X_test, y_test )

            util.show_confusion_matrix(df1_train_models, X_test, y_test)

            util.show_ROC_plots({"Logistic_Regression": best_model}, X_test, y_test)

            util.precision_recall_threshold(best_model, X_test,y_test)


    # test against our test data
    # util.analyze_models(models)

###########################################################################
## ANN SECTION ############################################################
ANN = False
if ANN == True:
    train_baseline = True
    ######### STEP 1
    # Data already split for test and validation, training is external 
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # read in dataset1.csv
    df1 = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-training_timeline1.txt")

    df2 = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-test_timeline1.txt")


    # remove any columns that are not in both datasets
    util.synch_datasets(df1, df2)

    # Set weights for training
    n_class_1 = df1_train_ann[df1_train['tag'] == 1].shape[0]
    n_class_0 = df1_train_ann.shape[0] - n_class_1
    n_samples = df1_train_ann.shape[0]

    # Set class weights for training
    class_weight = {
        0: n_samples / (2 * n_class_0),
        1: n_samples / (2 * n_class_1)
    }

    print("\n Step 1 complete. \n")
    ######### STEP 2
    print("\n Start step 2 - Performance metric selection. \n")
    # other settings
    last_layer_activation = 'sigmoid'
    optimizer = Adam(learning_rate=0.0001)
    loss_fn = 'binary_crossentropy'
    # loss_fn = 'catagorical_crossentropy'
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    print("\n Step 2 complete. \n")

    ######### STEP 3
    # further split our test data into validation data as well    
    # X_train, X_Val, y_train, y_Val = train_test_split(data_train_val_df, y, test_size=0.33, random_state=42)
    # split again to make half 15% test and 15% validation
    # print("\n Startin step 3 - Split the data. \n")
    # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    # print("\n Step 3 complete. \n")

    ######### STEP 4
    print("\n Startin step 4 - Data transforms. \n")

    # exclude_lst = []
    # nom_features_lst = ['sex', 'class_', 'deck', 'embark_town', 'alone']
    # num_features_lst = ['age', 'n_siblings_spouses', 'parch', 'fare']

    # X_train_rdy = transform_titanic_data(x_train, 
    #                                     exclude_list = exclude_lst, 
    #                                     nom_feat = nom_features_lst,
    #                                     num_feat = num_features_lst).astype('float64')

    # X_val_rdy = transform_titanic_data(x_val, 
    #                                     exclude_list = exclude_lst, 
    #                                     nom_feat = nom_features_lst,
    #                                     num_feat = num_features_lst).astype('float64')

    # X_test_rdy =transform_titanic_data(x_test, 
    #                                     exclude_list = exclude_lst, 
    #                                     nom_feat = nom_features_lst,
    #                                     num_feat = num_features_lst).astype('float64')
    # print("\n Step 4 complete. \n")

    ######### STEP 5 BASELINE MODEL
    print("\n Startin step 5 - Initial model \n")
    # if os.path.isdir("step_5_initial_model_titanic"):
    #     model4 = tf.keras.models.load_model('step_5_initial_model_titanic')

    #     pred_train= model4.predict(X_train_rdy)
    #     scores = model4.evaluate(X_train_rdy, y_train, verbose=0)
    #     print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

    #     pred_test= model4.predict(X_val_rdy)
    #     scores2 = model4.evaluate(X_val_rdy, y_val, verbose=0)
    #     print('Accuracy on validate data: {}% \n Error on validate data: {}'.format(scores2[1], 1 - scores2[1]))    
    # Set Xs and Ys for our training set

    training = True
    if training:
        X = df1_train_ann.drop(["datetime", "tag"], axis=1)
        y = df1_train_ann.tag
    else:
        X = df1_validation_ann.drop(["datetime", "tag"], axis=1)
        y = df1_validation_ann.tag

    # Get feature names
    feature_names = X.columns.values.tolist()

    # Build or load models
    load_models = True
    if load_models:
        if training:
            df1_train_models = util.load_all_models("training")
        else:
            df1_train_models = util.load_all_models("test")
    else:
        df1_train_models_ann = util.build_ann_classifiers(X, y, class_weight, save_models=True, dataset_used="training")

