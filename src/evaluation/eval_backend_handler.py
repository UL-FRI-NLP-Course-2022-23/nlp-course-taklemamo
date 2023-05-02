import json
from dataclasses import dataclass
import requests

import pandas as pd


TEXT_URL = "https://nlpprojtest.azurewebsites.net/api/text?code=deOJXHOXVtNr2Udl029XrFsDeZRexC2KVyMvpQr-JJK8AzFuiwmqEA%3D%3D"
SCORE_URL = "https://nlpprojtest.azurewebsites.net/api/scores?code=qG0eX5RGKDDnIyEGXPSPk3KKzMOPXl7ZhNQy_dgaokthAzFu6wfDyQ%3D%3D"


@dataclass
class TextSchema:
    textId: str
    originalText: str
    paraphrasedText: str


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
    res = requests.post(url=TEXT_URL, json=text_json)

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
            s_id = score_json["scorerId"]
            # add all or none (ugly, but since there are so few imperfect, cheaper than checking if keys exist)
            score_dict["appropriateness"].append(ap)
            score_dict["fluency"].append(fl)
            score_dict["diversity"].append(di)
            index.append(s_id)
        except KeyError:
            # skip invalid records
            pass

    df = pd.DataFrame(score_dict, index=index)
    return df


if __name__ == "__main__":
    # upload_text("test_py5", "original", "para")
    print(get_score_pd())
