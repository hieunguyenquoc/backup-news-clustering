from pydantic import BaseSettings


class Settings(BaseSettings):
    PORT = 2360
    HOST = "http://localhost:{}/".format(PORT)

    DATA_UPLOAD = "./data_upload/"
    PRETRAIN_MODEL = "./pretrained/"