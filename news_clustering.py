import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from config import Settings
import time

settings = Settings()

def load_model():
    """
    Load Sentence transformer model
    :return:
        model
    """
    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    return model


def get_embeddings(model, sentences):
    st = time.time()
    print("Getting embedding for document.")
    sentences_tokenizer = [tokenize(sentence) for sentence in sentences]
    embeddings = model.encode(sentences_tokenizer)
    print("Embedding time: ", time.time() - st)

    return embeddings


def fit_data(embeddings, df_sen, n_cluster, class_df):
    centers = np.zeros((n_cluster, embeddings.shape[1]))
    for cls, i in zip(class_df, range(n_cluster)):
        _s = df_sen[df_sen['cls'] == cls].index[0]
        _e = df_sen[df_sen['cls'] == cls].index[-1]
        Xk = embeddings[_s:_e + 1]
        centers[i, :] = np.mean(Xk, axis=0)
    return centers


def predict(model, df):
    X = get_embeddings(model, df['content'])
    with open(settings.PRETRAIN_MODEL + '/' + 'news_pretrain.npy', 'rb') as f:
        centers = np.load(f)
    class_predict = np.argmax(cosine_similarity(X, centers), axis=1)
    class_score = np.max(cosine_similarity(X, centers), axis=1)

    return class_predict, class_score


def evaluate(y_pred, y_true, label_unq):
    true = 0
    for i in range(len(y_pred)):
        if label_unq[y_pred[i]] == y_true[i]:
            true += 1

    return true / (len(y_pred))


def training(model, df):
    cls_df = df['cls'].unique()
    embeddings = get_embeddings(model, df['content'])
    if not os.path.exists(settings.PRETRAIN_MODEL + '/' + 'news_pretrain.npy'):
        pass
    else:
        with open(settings.PRETRAIN_MODEL + '/' + 'news_pretrain.npy', 'rb') as f:
            centers = np.load(f)
            embeddings = np.concatenate((embeddings, [centers]), axis=0)

    centers = fit_data(embeddings, df, len(cls_df), cls_df)
    # save centers to file numpy
    with open(settings.PRETRAIN_MODEL + '/' + 'news_pretrain.npy', 'wb') as f:
        np.save(f, centers)




