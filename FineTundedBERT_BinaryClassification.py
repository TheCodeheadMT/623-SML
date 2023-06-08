import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import SMLFinalPreprocessor as util


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print(tf.__version__)

model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = TFDistilBertForSequenceClassification.from_pretrained(model_path, id2label={0: "NEG", 1: "POS"},
                                                              label2id={"NEG": 0, "POS": 1})

# load data
PREPROCESS = True
if PREPROCESS:
    CUR_SET = "training"

    # read in dataset1.csv

    df_trn = pd.read_csv("MANUAL_LABELED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK_training_timeline1.txt",
                         header=0,
                         names=['datetime', 'timestamp_desc', 'source', 'source_long',
                                'message', 'parser', 'display_name', 'tag'])

    df_tst = pd.read_csv("MANUAL_LABELED_2023-02-17_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK_test_timeline1.txt", header=0,
                         names=['datetime', 'timestamp_desc', 'source', 'source_long',
                                'message', 'parser', 'display_name', 'tag'])

    df_trn_ = df_trn.drop(['datetime', 'display_name'], axis=1)
    df_tst_ = df_tst.drop(['datetime', 'display_name'], axis=1)

    df_trn_ = util.remove_path_chars(df_trn_)
    df_tst_ = util.remove_path_chars(df_tst_)



    df_trn_.to_csv("BERT_DATA_trn.csv")
    df_tst_.to_csv("BERT_DATA_tst.csv")

    exit()

TRAIN = True
if TRAIN:
    df_trn = pd.read_csv("BERT_DATA_trn.csv")
    df_tst = pd.read_csv("BERT_DATA_tst.csv")

    df_trn_msg_tag = df_trn[['message', 'tag']]
    x_test = df_tst[['message', 'tag']]

    # split data
    x_train, x_valid = train_test_split(df_trn_msg_tag, test_size=0.25, shuffle=True, stratify=df_trn_msg_tag['tag'])

    # Shapes
    # print(x_train.shape)   (69573, 7)
    # print(x_valid.shape)   (23192, 7)
    # print(df_tst.shape)    (6184, 7)
    # x_train = Dataset.from_pandas(x_train)
    # x_valid = Dataset.from_pandas(x_valid)

    tokens_trn = [str(i) for i in x_train['message'].values]
    tokens_tst = [str(i) for i in x_test['message'].values]
    tokens_val = [str(i) for i in x_valid['message'].values]
    # create tokenizer
    enc_train = tokenizer(tokens_trn, padding=True, truncation=True)
    enc_val = tokenizer(tokens_val, padding=True, truncation=True)
    # enc_test = tokenizer(tokens_tst, padding=True, truncation=True)

    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(enc_train), list(x_train.tag)))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(enc_val), list(x_valid.tag)))
    # test_dataset = tf.data.Dataset.from_tensor_slices((dict(enc_test), list(x_test.tag)))

    # model summary
    model.summary()

    ############################
    # fit callbacks
    CB_EARLYSTOP = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000000000001, patience=5)
    CB_TENSORBOARD = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    # Set class weights for training
    n_class_1 = df_trn[df_trn['tag'] == 1].shape[0]
    n_class_0 = df_trn.shape[0] - n_class_1
    n_samples = df_trn.shape[0]

    CLASS_WEIGHT = {
        0: n_samples / (2 * n_class_0),
        1: n_samples / (2 * n_class_1)
    }

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='balanced_accuracy'),
        #tf.keras.metrics.Precision(name='precision'),
        #tf.keras.metrics.Recall(name='recall'),
        #tf.keras.metrics.AUC(name='prc', curve='PR')
    ]

    # HYPER PARAMETERS, LOSS, and OPTIMIZATION
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=5e-5)
    #LOSS = tf.keras.losses.BinaryCrossentropy()
    LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    BATCH_SIZE = 16
    EPOCHS = 3

    # build the model
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=METRICS)

    history = model.fit(train_dataset.shuffle(len(x_train)).batch(BATCH_SIZE),
                        validation_data=val_dataset.shuffle(len(x_valid)).batch(BATCH_SIZE),
                        class_weight=CLASS_WEIGHT,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[CB_TENSORBOARD, CB_EARLYSTOP])

    model.save('bert_models/Fine_Tuned_BERT_BinaryClass_initial.h5')
else:
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path, id2label={0: "NEG", 1: "POS"},
                                                                     label2id={"NEG": 0, "POS": 1})
    model.load_weights('bert_models/model_reg_D32tan_D16tan_D1sig.h5')



REPORT = False
if REPORT:
    class_names = ['system', 'user']

    # Test the best model
    best_model = model

    y_pred_test = model.predict(enc_train)

    # using threshold 0.5
    y_pred_test[y_pred_test > 0.5] = 1
    y_pred_test[y_pred_test < 0.5] = 0
    print(classification_report(y_train, y_pred_train, target_names=class_names))

    model_CM = confusion_matrix(y_pred=y_pred_train, y_true=y_train)

    util.plot_confusion_matrix(model_CM, classes=class_names,
                          title='Confusion matrix, without normalization - train')
    plt.savefig('bert_figures/Confusion matrix, without normalization')

    util.plot_confusion_matrix(model_CM, classes=class_names, title='Confusion matrix (norm) - train')
    plt.savefig('bert_figures/Confusion matrix (norm)')

    #find_perf_precision(model, X_train, y_train, y_pred_train)
    RocCurveDisplay.from_predictions(y_train, model.predict(X_train), name="ROC Training")
    plt.savefig("bert_figures/roc_curve_training.png")

