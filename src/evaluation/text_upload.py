import requests


TEXT_URL = "https://nlpprojtest.azurewebsites.net/api/text?code=deOJXHOXVtNr2Udl029XrFsDeZRexC2KVyMvpQr-JJK8AzFuiwmqEA%3D%3D"


class TextSchema:
    def __init__(self, textId, originalText, paraphrasedText):
        self.textId = textId
        self.originalText = originalText
        self.paraphrasedText = paraphrasedText


def upload_text(id, original, paraphrase):
    text_obj = TextSchema(id, original, paraphrase)
    text_json = text_obj.__dict__
    res = requests.post(url=TEXT_URL, json=text_json)

    if res.status_code != 201:
        raise Exception(f"Failed to upload text, reason: {res.status_code} - {res.content}")


if __name__ == "__main__":
    upload_text("test_py5", "original", "para")
