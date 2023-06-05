import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import SMLFinalPreprocessor as util


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print(tf. __version__)

model_path= 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = TFDistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})

# load data
PREPROCESS = True
if PREPROCESS:
    CUR_SET = "training"

    # read in dataset1.csv

    df_trn = pd.read_csv("MANUAL_LABELED_2023-02-09_LVL5-FILTER_FILE-OLECF-WEBHIST-LNK_training_timeline1.txt", header=0,
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
    x_train, x_valid = train_test_split(df_trn_msg_tag, test_size=0.25, shuffle= True, stratify=df_trn_msg_tag['tag'])

    # Shapes
    #print(x_train.shape)   (69573, 7)
    #print(x_valid.shape)   (23192, 7)
    #print(df_tst.shape)    (6184, 7)
    #x_train = Dataset.from_pandas(x_train)
    #x_valid = Dataset.from_pandas(x_valid)

    # create tokenizer
    enc_train = tokenizer(list(x_train['message'].values), padding=True, truncation=True)
    #enc_test = x_test.apply(lambda x: tokenizer(x['message'], padding=True, truncation=True, batched=True, batch_size=1000))
    #enc_val = x_valid.apply(lambda x: tokenizer(x['message'], padding=True, truncation=True, batched=True, batch_size=1000))

    pd.DataFrame(enc_train)