import numpy as np

from news_clustering import load_model, training, predict
from typing import Optional
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from models import ListNews
from utils import parse_input_json_dataframe

app = FastAPI(docs_url='/docs')


model = load_model()

@app.post("/post_news")
def _post_new(data: ListNews):
    df = parse_input_json_dataframe(data)
    res_pred, res_score = predict(model, df)
    res_pred = res_pred.tolist()
    res_score = res_score.tolist()
    data = []
    for i in range(len(df)):
        data.append({"id":df.iloc[i]['id'], "pred_cls":res_pred[i], "score":res_score[i]})
    return {
        'data': data
    }


@app.post("/input_train_data")
def _input_train_data(data: ListNews):
    df = parse_input_json_dataframe(data)
    res = training(model, df)
    return True


