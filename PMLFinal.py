from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, RocCurveDisplay, precision_recall_curve
from sklearn import metrics
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from sklearn.pipeline import Pipeline
from IPython.display import SVG
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import SMLFinalPreprocessor as util
import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import itertools



# Initial config:
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def tune_final(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources={"cpu": 2, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="balanced_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=10,
        ),
        run_config=air.RunConfig(
            name="exp",
            stop={"balanced_accuracy": 0.99, "training_iteration": num_training_iterations},
        ),
        param_space={
            "threads": 2,
            "lr": tune.uniform(0.0001, 0.01),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(32, 512),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

def train_final(config, ):
    batch_size = 128
    num_classes = 1
    epochs = 10
    feature_size = X_train.shape[1]

    metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]


    with FileLock(os.path.expanduser("data.lock")):
        df1_train, df2_test, X_train, y_train, X_valid, y_valid, X_test, y_test, class_weight = load_dataset()

    
    #n_epochs = 10
    #batch_size = 128
    #optimizer = RMSprop(learning_rate=1e-5)
    #loss = 'binary_crossentropy'

    input_tensor = tf.keras.layers.Input(shape=(feature_size))
    current_tensor = input_tensor
    current_tensor = BatchNormalization()(current_tensor)
    current_tensor = Dense(32, activation='tanh')(current_tensor)
    current_tensor = Dense(16, activation='tanh')(current_tensor)
    current_tensor = Dense(num_classes, activation='sigmoid')(current_tensor)
    output_tensor = current_tensor
    model = Model(input_tensor, output_tensor)
 
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(lr=config["lr"], momentum=config["momentum"]),
        metrics=metrics
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({"balanced_accuracy": "bal_acc"})],
    )

def get_model(input_shape, overfit=False, reg=False):
    num_classes = 1
    feature_size = input_shape

    metrics = [
    tf.keras.metrics.BinaryAccuracy(name='balanced_accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='prc', curve='PR')]

    input_tensor = tf.keras.layers.Input(shape=(feature_size))
    current_tensor = input_tensor
    current_tensor = BatchNormalization()(current_tensor)
    current_tensor = Dense(32, activation='relu')(current_tensor)
    if reg:
        current_tensor = Dropout(.2)(current_tensor)
    if overfit:
        current_tensor = Dense(32, activation='relu')(current_tensor)
        if reg:
            current_tensor = Dropout(.2)(current_tensor)

    current_tensor = Dense(16, activation='relu')(current_tensor)
    if reg:
        current_tensor = Dropout(.2)(current_tensor)

    current_tensor = Dense(8, activation='relu')(current_tensor)

    current_tensor = Dense(num_classes, activation='sigmoid')(current_tensor)
    output_tensor = current_tensor
    model = Model(input_tensor, output_tensor)
 
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=metrics
    )

    return model
    
def load_dataset():
     # STEP 1 - LOAD DATA ##########################################################
    # read in dataset1.csv
    df1_train = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-training_timeline1.txt")

    df2_test = pd.read_csv(
        "MANUAL_LABELED_ENCODED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK-MSG_LEN-KEY_DIR-test_timeline1.txt")

    # remove any columns that are not in both datasets
    util.synch_datasets(df1_train, df2_test)

    # split data for training and validation
    #c1_non_test_df, c1_test_df = train_test_split(c1_df, test_size=.25, random_state=42, stratify=c1_df['Class'])
    x_train, x_valid = train_test_split(df1_train, test_size=0.25 , shuffle= True, stratify=df1_train['tag'])

    # setup training data
    X_train = x_train.drop(['tag', 'datetime'], axis=1)
    y_train = x_train['tag']

    # setup validation data
    X_valid = x_valid.drop(['tag', 'datetime'], axis=1)
    y_valid = x_valid['tag']

    # setup test data
    X_test = df2_test.drop(['tag', 'datetime'], axis=1)
    y_test = df2_test['tag']

    # Set class weights for training
    n_class_1 = df1_train[df1_train['tag'] == 1].shape[0]
    n_class_0 = df1_train.shape[0] - n_class_1
    n_samples = df1_train.shape[0]
    
    class_weight = {
        0: n_samples / (2 * n_class_0),
        1: n_samples / (2 * n_class_1)
    }

    return df1_train, df2_test, X_train, y_train, X_valid, y_valid, X_test, y_test, class_weight 


def final():

    # load data
    df1_train, df2_test, X_train, y_train, X_valid, y_valid, X_test, y_test, class_weight = load_dataset()
    
    TRAIN = True
    if TRAIN:
        # get model
        model = get_model(input_shape=X_train.shape[1], overfit=False, reg=True)
        
        model.summary()

        #plot_model(model, "pml_figures/reg_model.png")

        # fit callbacks
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000000000001, patience=5)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # mode fit config items
        batch_size = 128
        epochs = 50

        history = model.fit(x=X_train,
                                y=y_train,
                                validation_data=(X_valid, y_valid),
                                class_weight=class_weight,
                                batch_size=batch_size,
                                shuffle=True,
                                epochs=epochs,
                                callbacks=[tensorboard_callback, earlystop])

        model.save('models/model_reg_D32tan_D16tan_D1sig.h5')

        print("Baseline model\n")
        
    else:

        model = get_model(input_shape=X_train.shape[1], overfit=False, reg=True)
        model.load_weights('models/model_reg_D32tan_D16tan_D1sig.h5')
    
    REPORT_RESULTS=True
    if REPORT_RESULTS:
        
        class_names = ['system', 'user']
        y_pred_train = model.predict(X_train)
        y_pred_val= model.predict(X_valid)
        
        
        # using threshold 0.5
        y_pred_train[y_pred_train > 0.5] = 1
        y_pred_train[y_pred_train < 0.5] = 0
        y_pred_val[y_pred_val > 0.5] = 1
        y_pred_val[y_pred_val < 0.5] = 0
        

        scores = model.evaluate(X_train, y_train, verbose=0)
        bal_acc_train = balanced_accuracy_score(y_train, y_pred_train)

        print('Accuracy on training data: {}%, {}(balanaced) \n Error on training data: {}'.format(scores[1], bal_acc_train, 1 - scores[1]))   
        print('\n')  

        # #cross val score
        # scoring = 'balanced_accuracy'
        # kfold = KFold(n_splits=10)
        # cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring ='accuracy')
        # results.append(cv_results)
        # #names.append(name)
        # msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
        # print(msg)

        # #pred = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy')
        # #mean_cv_accuracy = np.mean(pred)
        # print("Mean accuracy CV : " + str(mean_cv_accuracy) + "\n")    
        # exit()

        print(classification_report(y_train, y_pred_train, target_names=class_names))

        model_CM = confusion_matrix(y_pred=y_pred_train, y_true=y_train)                                                            

        plot_confusion_matrix(model_CM, classes=class_names, 
                            title='Confusion matrix, without normalization - train')
        plt.savefig('pml_figures/Confusion matrix, without normalization')

        plot_confusion_matrix(model_CM, classes=class_names, title='Confusion matrix (norm) - train')
        plt.savefig('pml_figures/Confusion matrix (norm)')

        #find_perf_precision(model, X_train, y_train, y_pred_train)
        RocCurveDisplay.from_predictions(y_train, model.predict(X_train), name="ROC Training")
        plt.savefig("pml_figures/roc_curve_trainin.png")

        precision_recall_threshold(model, X_train, y_train)

        scores2 = model.evaluate(X_valid, y_valid, verbose=0)
        bal_acc_valid = balanced_accuracy_score(y_valid, y_pred_val)
        print('Accuracy on validate data: {}%, {}(balanaced) \n Error on validate data: {}'.format(scores2[1], bal_acc_valid, 1 - scores2[1])) 
        print('\n')  
        
        print(classification_report(y_valid, y_pred_val, target_names=class_names))

        model_CM = confusion_matrix(y_pred=y_pred_val, y_true=y_valid)                                                            

        plt.figure()
        plot_confusion_matrix(model_CM, classes=class_names, 
                            title='Confusion matrix, without normalization - Training')
        plt.savefig('pml_figures/Test_Confusion_matrix_no_norm', bbox_inches="tight")
        
        plt.figure()
        plot_confusion_matrix(model_CM, normalize=True, classes=class_names, title='Confusion matrix (norm) - Training')
        plt.savefig('pml_figures/Test_Confusion_matrix_norm', bbox_inches="tight")



    TEST_WITH_BEST_MODEL = True
    if TEST_WITH_BEST_MODEL:
        print('Testing best model:')
        class_names = ['system', 'user']

        # Test the best model
        best_model = model
        
        y_pred_test= model.predict(X_test)

        # using threshold 0.5
        y_pred_test[y_pred_test > 0.5] = 1
        y_pred_test[y_pred_test < 0.5] = 0

        # print prarms:
        # for parameter in best_model.get_params():
        #     print(parameter, ': \t', best_model.get_params()[parameter])


        scores3 = model.evaluate(X_test, y_test, verbose=0)
        bal_acc_test = balanced_accuracy_score(y_test, y_pred_test)
        print('Accuracy on Test data: {}%, {}(balanaced) \n Error on Test data: {}'.format(scores3[1], bal_acc_test, 1 - scores3[1])) 
        print('\n')  
        
        print(classification_report(y_test, y_pred_test, target_names=class_names))

        model_CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)                                                            
        
        plt.figure()
        plot_confusion_matrix(model_CM, classes=class_names, 
                            title='Confusion matrix, without normalization - Test')
        plt.savefig('pml_figures/Test_Confusion_matrix_no_norm', bbox_inches="tight")
        
        plt.figure()
        plot_confusion_matrix(model_CM, normalize=True, classes=class_names, title='Confusion matrix (norm) - Test')
        plt.savefig('pml_figures/Test_Confusion_matrix_norm', bbox_inches="tight")


def find_perf_precision(model, X_train, y_train, y_pred_train):

    # predict probabilities
    yhat = y_pred_train

    # get false pos rate, true pos rate and thresholds
    fpr, tpr, thresholds = roc_curve(y_train, yhat)

    #obtain prediction probs on test set using the model fit previously on the non-test data
    preds_test = model.predict(X_train)

    # place holder
    threshold_perfect_precision = 0
    threshold_1 = -1

    #classify the prediction probabilities
    for idx, thld in enumerate(thresholds):
        y_hat = (preds_test>=thld)*1.0
        predPos = y_hat==1 
        
        truePos = np.zeros_like(y_hat)
        y_train1 = y_train==1
        for idx in tqdm(range(truePos.shape[1] // 1000), desc="Predicting"):
            slicer = slice(idx * 1000, (idx + 1) * 1000)
            cur_chunk_pred = predPos[slicer]
            cur_chunk_ytrain = y_train1[slicer]
            truePos[slicer] = np.logical_and(cur_chunk_pred,cur_chunk_ytrain)
            
        
        predPos1s = predPos*1.0
        if sum(predPos1s)[0] > 0:
            print(sum(predPos1s)[0])
            prec = sum(truePos*1.0)/sum()
            if prec[0] == 1:
                print(f'Found a perfect precision threshold!-> {thld} at idx {idx}')
                threshold_1 = thld
        
            
   

    print("Found  threshold of ", threshold_1 , " for precision 1.")

def precision_recall_threshold(model, X, y):
    test_x = X
    test_y = y

    pred_y=model.predict(test_x)
    probs_y=pred_y

    # using threshold 0.5
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y < 0.5] = 0

    precision, recall, thresholds = precision_recall_curve(test_y, probs_y)
    #retrieve probability of being 1(in second column of probs_y)
    pr_auc = metrics.auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.savefig("pml_figures/precision_recall_curve.png")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    this function is from https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html

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
    plt.xticks(tick_marks, classes)
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

if __name__ == "__main__":
    final()