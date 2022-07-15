import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import time
import re
import six
import json


def preprocess(text):
    """
    Preprocess input text
    :param text:
    :return:
    """
    text = str(text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'AT_ABC', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'([\w]+\) )', '.', text)
    text = re.sub(r'([\d]+\. )', '', text)
    text = re.sub(
        r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0-9., ]',
        '', text)
    #   text = text.replace('.', '. ')
    #   text = text.replace(' .', '. ')
    text = re.sub(r'[\s]+', ' ', text)
    text = re.sub(r'[.]+', '.', text)
    text = text.replace('. . ', '. ')
    text = text.strip('.')
    text = text.strip('\'"')
    text = text.strip()
    #   text = text.lower()
    return text


def load_csv(filename):
    """
    Load csv to dataframe
    :param
        filename: path to csv file
    :return:
        datarame: ['content', 'class']
    """
    df = pd.read_csv(filename)
    df.dropna(subset=['long_content'], inplace=True)
    df_sen = pd.DataFrame(columns=['content', 'class'])
    df_sen['content'] = df['title'] + '. ' + df['long_content']
    df_sen.head()
    df_sen['class'] = df['article_type']
    df_sen = df_sen.reset_index(drop=True)
    df_sen.sort_values(by=['class'])

    class_df = df_sen['class'].unique()

    return df_sen, class_df


def parse_input_json_dataframe(data: str):
    df = pd.DataFrame(columns=['id', 'title', 'content', 'cls'])
    # print("len:", len(data))
    for news in data['data']:
        # print(news.id)
        # df = df.append({'id': news.id, 'pred_cls': 'abc', 'score': 1.}, ignore_index=True)
        df = df.append({'id':news.id, 'title':news.title, 'content':news.content, 'cls':news.cls}, ignore_index=True)
    print(df)
    return df

