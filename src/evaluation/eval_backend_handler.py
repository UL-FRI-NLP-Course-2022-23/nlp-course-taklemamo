import json
from dataclasses import dataclass
import requests

import pandas as pd
import numpy as np


TEXT_POST_URL = "https://nlpprojtest.azurewebsites.net/api/text?code=deOJXHOXVtNr2Udl029XrFsDeZRexC2KVyMvpQr-JJK8AzFuiwmqEA%3D%3D"
TEXT_GET_URL = "https://nlpprojtest.azurewebsites.net/api/texts/{}?code=csyFDq1DGzghi5_H36oOUtda68CVDrE6WBtYphM6uLHLAzFu1sEVAw%3D%3D"
SCORE_URL = "https://nlpprojtest.azurewebsites.net/api/scores?code=qG0eX5RGKDDnIyEGXPSPk3KKzMOPXl7ZhNQy_dgaokthAzFu6wfDyQ%3D%3D"


@dataclass
class TextSchema:
    textId: str
    originalText: str
    paraphrasedText: str

    @classmethod
    def from_json(cls, json_obj):
        # remove mongo stuff
        del json_obj["_id"]
        del json_obj["__v"]

        return cls(**json_obj)


@dataclass()
class ScoreSchema:
    textId: str
    scorerId: str
    appropriateness: float
    fluency: float
    diversity: float
    overall: float

    @classmethod
    def from_json(cls, json_obj):
        # remove mongo stuff
        del json_obj["_id"]
        del json_obj["__v"]

        return cls(**json_obj)


def upload_text(id, original, paraphrase):
    text_obj = TextSchema(id, original, paraphrase)
    text_json = text_obj.__dict__
    res = requests.post(url=TEXT_POST_URL, json=text_json)

    if res.status_code != 201:
        raise Exception(f"Failed to upload text, reason: {res.status_code} - {res.text}")


def get_score_objs():
    res = requests.get(SCORE_URL)
    if res.status_code != 200:
        raise Exception(f"Failed getting scores, reason: {res.status_code} - {res.text}")

    score_json_list = json.loads(res.text)

    score_objs = []
    for json_obj in score_json_list:
        try:
            score_objs.append(ScoreSchema.from_json(json_obj))
        except TypeError:
            # skip if no match with schema
            pass

    return score_objs


def get_score_pd():
    res = requests.get(SCORE_URL)
    if res.status_code != 200:
        raise Exception(f"Failed getting scores, reason: {res.status_code} - {res.text}")

    score_json_list = json.loads(res.text)

    score_dict = {
        "textId": [],
        "scorerId": [],
        "appropriateness": [],
        "fluency": [],
        "diversity": []
    }
    index = []
    for score_json in score_json_list:
        try:
            ap = float(score_json["appropriateness"])
            fl = float(score_json["fluency"])
            di = float(score_json["diversity"])
            
            # add all or none (ugly, but since there are so few imperfect, cheaper than checking if keys exist)
            score_dict["appropriateness"].append(ap)
            score_dict["fluency"].append(fl)
            score_dict["diversity"].append(di)

            score_dict["textId"].append(score_json["textId"])
            score_dict["scorerId"].append(score_json["scorerId"])
        except KeyError:
            # skip invalid records
            pass

    df = pd.DataFrame(score_dict)
    return df


def upload_random_text(n=10):
    datadir = "data/backtranslate/"
    readpath = datadir + "backtranslate_filtered.csv"

    df = pd.read_csv(readpath, header=None, sep='\t', names=['s', 'p'])
    df = df[df['s'].str.len() < 200]

    sentences = df['s'].tolist()
    paraps = df['p'].tolist()

    rnd = np.random.choice(len(sentences), n, replace=False)
    id = "naduskladitev"
    for i, rnd_index in enumerate(rnd):
        upload_text(f"{id}-{i}", sentences[rnd_index], paraps[rnd_index])


def get_text(id):
    res = requests.get(TEXT_GET_URL.format(id))
    if res.status_code != 200:
        raise Exception(f"Failed getting text, reason: {res.status_code} - {res.text}")

    text_json = json.loads(res.text)[0]
    return TextSchema.from_json(text_json)


if __name__ == "__main__":
    # upload_text("test_py5", "original", "para")
    # print(get_score_pd())
    # upload_random_text()
    print(get_text("naduskladitev-0"))
